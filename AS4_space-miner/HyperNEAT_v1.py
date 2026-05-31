import argparse
import copy
import json
import math
import os
import pickle
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # numpy is optional; the script falls back to pure Python.
    np = None


SCRIPT_DIR = Path(__file__).resolve().parent


DEFAULT_CONFIG = {
    "environment": {
        "width": 800,
        "height": 600,
        "ship_radius": 15,
        "ship_speed": 5.0,
        "max_turn_rate": 0.20,
        "fuel_capacity": 100.0,
        "fuel_cost_per_thrust": 0.12,
        "fuel_cost_per_step": 0.0,
        "fuel_gain_per_mineral": 10.0,
        "initial_minerals": 5,
        "min_minerals": 3,
        "mineral_radius": 10,
        "asteroid_count": 8,
        "asteroid_radius_min": 15,
        "asteroid_radius_max": 30,
        "asteroid_speed_min": 0.20,
        "asteroid_speed_max": 2.00,
        "max_steps": 1200,
        "mine_threshold": 0.50,
        "thrust_deadzone": 0.05,
        "idle_thrust_threshold": 0.05,
        "spawn_margin": 20,
        "max_spawn_attempts": 100,
    },
    "grid": {
        "size": 9,
        "view_radius": 280.0,
        "mineral_signal_radius": 65.0,
        "asteroid_signal_radius": 95.0,
    },
    "substrate": {
        "hidden_grid_size": 7,
        "weight_threshold": 0.30,
        "weight_scale": 3.0,
        "include_direct_input_output": False,
        "activation": "tanh",
        "use_numpy": True,
    },
    "neat": {
        "generations": 25,
        "pop_size": 60,
        "fitness_threshold": 10000.0,
        "reset_on_extinction": True,
        "num_cppn_inputs": 8,
        "num_cppn_outputs": 1,
        "initial_hidden": 0,
        "initial_connection": "full_direct",
        "activation_default": "tanh",
        "activation_options": "tanh sigmoid gauss sin",
        "activation_mutate_rate": 0.10,
        "aggregation_default": "sum",
        "aggregation_options": "sum",
        "aggregation_mutate_rate": 0.0,
        "bias_init_mean": 0.0,
        "bias_init_stdev": 1.0,
        "bias_replace_rate": 0.10,
        "bias_mutate_rate": 0.70,
        "bias_mutate_power": 0.50,
        "bias_max_value": 30.0,
        "bias_min_value": -30.0,
        "response_init_mean": 1.0,
        "response_init_stdev": 0.10,
        "response_replace_rate": 0.10,
        "response_mutate_rate": 0.10,
        "response_mutate_power": 0.10,
        "response_max_value": 30.0,
        "response_min_value": -30.0,
        "weight_max_value": 30.0,
        "weight_min_value": -30.0,
        "weight_init_mean": 0.0,
        "weight_init_stdev": 1.0,
        "weight_mutate_rate": 0.20,
        "weight_replace_rate": 0.20,
        "weight_mutate_power": 0.50,
        "conn_add_prob": 0.08,
        "conn_delete_prob": 0.04,
        "node_add_prob": 0.06,
        "node_delete_prob": 0.03,
        "enabled_default": True,
        "enabled_mutate_rate": 0.05,
        "compatibility_disjoint_coefficient": 1.0,
        "compatibility_weight_coefficient": 0.6,
        "compatibility_threshold": 3.1,
        "species_fitness_func": "mean",
        "max_stagnation": 15,
        "species_elitism": 2,
        "elitism": 3,
        "survival_threshold": 0.30,
        "stdout_reporter": True,
    },
    "fitness": {
        "mineral_reward": 150.0,
        "progress_reward": 30.0,
        "alive_reward": 0.02,
        "fuel_remaining_reward": 10.0,
        "collision_penalty": 150.0,
        "out_of_fuel_penalty": 80.0,
        "empty_mine_penalty": 0.80,
        "idle_penalty": 0.02,
        "danger_penalty": 0.50,
    },
    "seeds": {
        "training": [101, 202, 303],
        "evaluation": [404, 505, 606, 707, 808],
    },
    "outputs": {
        "winner_path": "hyperneat_v1_winner.pkl",
        "metrics_path": "hyperneat_v1_metrics.json",
    },
    "render": {
        "winner": False,
        "seed": 404,
        "fps": 60,
        "show_grid": True,
    },
}


GRID_CHANNELS = ("mineral", "asteroid_danger", "asteroid_vx", "asteroid_vy")
GLOBAL_INPUTS = ("fuel", "heading_cos", "heading_sin", "bias")
OUTPUT_NAMES = ("heading_x", "heading_y", "thrust", "mine")


@dataclass(frozen=True)
class SubstrateNode:
    x: float
    y: float
    z: float
    name: str


@dataclass
class Body:
    x: float
    y: float
    radius: float
    speed_x: float = 0.0
    speed_y: float = 0.0


@dataclass
class Ship:
    x: float
    y: float
    radius: float
    angle: float = 0.0
    fuel: float = 100.0
    minerals: int = 0


@dataclass
class EpisodeMetrics:
    fitness: float
    minerals: int
    steps: int
    fuel_remaining: float
    collision: bool
    out_of_fuel: bool
    empty_mine_attempts: int
    idle_frames: int
    progress: float
    danger: float


def deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as config_file:
            user_config = json.load(config_file)
        config = deep_merge(config, user_config)
    validate_config(config)
    return config


def validate_config(config):
    grid_size = int(config["grid"]["size"])
    hidden_grid_size = int(config["substrate"]["hidden_grid_size"])
    if grid_size < 3 or grid_size % 2 == 0:
        raise ValueError("grid.size must be an odd integer >= 3")
    if hidden_grid_size < 1:
        raise ValueError("substrate.hidden_grid_size must be >= 1")
    if float(config["grid"]["view_radius"]) <= 0:
        raise ValueError("grid.view_radius must be positive")
    if not config["seeds"]["training"]:
        raise ValueError("seeds.training must contain at least one seed")
    if not config["seeds"]["evaluation"]:
        raise ValueError("seeds.evaluation must contain at least one seed")


def resolve_output_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return SCRIPT_DIR / path


def write_default_config(path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as config_file:
        json.dump(DEFAULT_CONFIG, config_file, indent=2)
        config_file.write("\n")
    print(f"Wrote default config to {output_path}")


def import_neat():
    try:
        import neat
    except ModuleNotFoundError as error:
        raise SystemExit(
            "The 'neat' package is required for training. Activate the project environment "
            "from AS4_space-miner/environment.yml or install neat-python==0.92."
        ) from error
    return neat


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def wrap_delta(delta, size):
    half = size / 2.0
    if delta > half:
        delta -= size
    elif delta < -half:
        delta += size
    return delta


def angle_delta(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


def normalized_positions(count):
    if count == 1:
        return [0.0]
    return [-1.0 + 2.0 * i / (count - 1) for i in range(count)]


class SpaceMinerEnv:
    def __init__(self, config):
        self.config = config
        self.env_cfg = config["environment"]
        self.grid_cfg = config["grid"]
        self.fitness_cfg = config["fitness"]
        self.width = float(self.env_cfg["width"])
        self.height = float(self.env_cfg["height"])
        self.rng = random.Random()
        self.ship = None
        self.minerals = []
        self.asteroids = []
        self.steps = 0
        self.done = False
        self.collision = False
        self.out_of_fuel = False
        self.empty_mine_attempts = 0
        self.idle_frames = 0
        self.progress_sum = 0.0
        self.danger_sum = 0.0
        self.fuel_used = 0.0

    def reset(self, seed):
        self.rng = random.Random(int(seed))
        self.ship = Ship(
            x=self.width / 2.0,
            y=self.height / 2.0,
            radius=float(self.env_cfg["ship_radius"]),
            angle=0.0,
            fuel=float(self.env_cfg["fuel_capacity"]),
        )
        self.steps = 0
        self.done = False
        self.collision = False
        self.out_of_fuel = False
        self.empty_mine_attempts = 0
        self.idle_frames = 0
        self.progress_sum = 0.0
        self.danger_sum = 0.0
        self.fuel_used = 0.0
        self.minerals = [self._spawn_mineral() for _ in range(int(self.env_cfg["initial_minerals"]))]
        self.asteroids = [self._spawn_asteroid() for _ in range(int(self.env_cfg["asteroid_count"]))]
        return self.observe()

    def _spawn_position(self, radius, avoid_ship=True):
        margin = max(float(self.env_cfg["spawn_margin"]), radius)
        max_attempts = int(self.env_cfg["max_spawn_attempts"])
        for _ in range(max_attempts):
            x = self.rng.uniform(margin, self.width - margin)
            y = self.rng.uniform(margin, self.height - margin)
            if not avoid_ship:
                return x, y
            if self._distance_to_ship(x, y) > self.ship.radius + radius + 80.0:
                return x, y
        return self.rng.uniform(margin, self.width - margin), self.rng.uniform(margin, self.height - margin)

    def _spawn_mineral(self):
        radius = float(self.env_cfg["mineral_radius"])
        x, y = self._spawn_position(radius)
        return Body(x=x, y=y, radius=radius)

    def _spawn_asteroid(self):
        radius = self.rng.uniform(
            float(self.env_cfg["asteroid_radius_min"]),
            float(self.env_cfg["asteroid_radius_max"]),
        )
        x, y = self._spawn_position(radius)
        speed_min = float(self.env_cfg["asteroid_speed_min"])
        speed_max = float(self.env_cfg["asteroid_speed_max"])
        speed = self.rng.uniform(speed_min, speed_max)
        heading = self.rng.uniform(0.0, math.tau)
        return Body(
            x=x,
            y=y,
            radius=radius,
            speed_x=math.cos(heading) * speed,
            speed_y=math.sin(heading) * speed,
        )

    def _relative_to_ship(self, x, y):
        return (
            wrap_delta(x - self.ship.x, self.width),
            wrap_delta(y - self.ship.y, self.height),
        )

    def _distance_to_ship(self, x, y):
        dx, dy = self._relative_to_ship(x, y)
        return math.hypot(dx, dy)

    def _nearest_mineral_distance(self):
        if not self.minerals:
            return None
        return min(self._distance_to_ship(mineral.x, mineral.y) for mineral in self.minerals)

    def _asteroid_danger_at_ship(self):
        signal_radius = float(self.grid_cfg["asteroid_signal_radius"])
        danger = 0.0
        for asteroid in self.asteroids:
            distance = self._distance_to_ship(asteroid.x, asteroid.y)
            edge_distance = max(0.0, distance - self.ship.radius - asteroid.radius)
            danger = max(danger, max(0.0, 1.0 - edge_distance / signal_radius))
        return danger

    def observe(self):
        grid_size = int(self.grid_cfg["size"])
        view_radius = float(self.grid_cfg["view_radius"])
        mineral_signal_radius = float(self.grid_cfg["mineral_signal_radius"])
        asteroid_signal_radius = float(self.grid_cfg["asteroid_signal_radius"])
        max_asteroid_speed = max(float(self.env_cfg["asteroid_speed_max"]), 1e-6)
        channel_count = len(GRID_CHANNELS)
        values = [0.0] * (grid_size * grid_size * channel_count)
        positions = normalized_positions(grid_size)

        mineral_offsets = [self._relative_to_ship(mineral.x, mineral.y) for mineral in self.minerals]
        asteroid_offsets = [self._relative_to_ship(asteroid.x, asteroid.y) for asteroid in self.asteroids]

        for row, y_norm in enumerate(positions):
            cell_y = y_norm * view_radius
            for col, x_norm in enumerate(positions):
                cell_x = x_norm * view_radius
                base_index = (row * grid_size + col) * channel_count

                mineral_signal = 0.0
                for mineral_dx, mineral_dy in mineral_offsets:
                    distance = math.hypot(cell_x - mineral_dx, cell_y - mineral_dy)
                    mineral_signal = max(mineral_signal, max(0.0, 1.0 - distance / mineral_signal_radius))

                danger_signal = 0.0
                velocity_x_signal = 0.0
                velocity_y_signal = 0.0
                for asteroid, (asteroid_dx, asteroid_dy) in zip(self.asteroids, asteroid_offsets):
                    distance = math.hypot(cell_x - asteroid_dx, cell_y - asteroid_dy)
                    edge_distance = max(0.0, distance - asteroid.radius)
                    danger = max(0.0, 1.0 - edge_distance / asteroid_signal_radius)
                    if danger > danger_signal:
                        danger_signal = danger
                        velocity_x_signal = clamp(asteroid.speed_x / max_asteroid_speed, -1.0, 1.0) * danger
                        velocity_y_signal = clamp(asteroid.speed_y / max_asteroid_speed, -1.0, 1.0) * danger

                values[base_index] = mineral_signal
                values[base_index + 1] = danger_signal
                values[base_index + 2] = velocity_x_signal
                values[base_index + 3] = velocity_y_signal

        fuel_ratio = self.ship.fuel / float(self.env_cfg["fuel_capacity"])
        values.extend(
            [
                clamp(fuel_ratio, 0.0, 1.0),
                math.cos(self.ship.angle),
                math.sin(self.ship.angle),
                1.0,
            ]
        )
        return values

    def step(self, action):
        if self.done:
            return self.observe(), True

        previous_distance = self._nearest_mineral_distance()
        previous_fuel = self.ship.fuel
        self.steps += 1

        desired_heading = float(action["desired_heading"])
        thrust = clamp(float(action["thrust"]), 0.0, 1.0)
        mine = bool(action["mine"])

        turn = clamp(
            angle_delta(desired_heading, self.ship.angle),
            -float(self.env_cfg["max_turn_rate"]),
            float(self.env_cfg["max_turn_rate"]),
        )
        self.ship.angle = (self.ship.angle + turn) % math.tau

        moved = False
        if self.ship.fuel > 0.0:
            step_fuel_cost = float(self.env_cfg["fuel_cost_per_step"])
            thrust_fuel_cost = float(self.env_cfg["fuel_cost_per_thrust"]) * thrust
            total_fuel_cost = step_fuel_cost + thrust_fuel_cost
            if total_fuel_cost > 0.0:
                self.ship.fuel = max(0.0, self.ship.fuel - total_fuel_cost)
            if thrust > float(self.env_cfg["thrust_deadzone"]) and self.ship.fuel > 0.0:
                distance = float(self.env_cfg["ship_speed"]) * thrust
                self.ship.x = (self.ship.x + math.cos(self.ship.angle) * distance) % self.width
                self.ship.y = (self.ship.y + math.sin(self.ship.angle) * distance) % self.height
                moved = True

        self.fuel_used += max(0.0, previous_fuel - self.ship.fuel)

        mined_count = 0
        if mine:
            remaining_minerals = []
            for mineral in self.minerals:
                if self._distance_to_ship(mineral.x, mineral.y) < self.ship.radius + mineral.radius:
                    mined_count += 1
                else:
                    remaining_minerals.append(mineral)
            self.minerals = remaining_minerals
            if mined_count:
                self.ship.minerals += mined_count
                self.ship.fuel = min(
                    float(self.env_cfg["fuel_capacity"]),
                    self.ship.fuel + float(self.env_cfg["fuel_gain_per_mineral"]) * mined_count,
                )
            else:
                self.empty_mine_attempts += 1

        while len(self.minerals) < int(self.env_cfg["min_minerals"]):
            self.minerals.append(self._spawn_mineral())

        for asteroid in self.asteroids:
            asteroid.x = (asteroid.x + asteroid.speed_x) % self.width
            asteroid.y = (asteroid.y + asteroid.speed_y) % self.height

        self.collision = any(
            self._distance_to_ship(asteroid.x, asteroid.y) < self.ship.radius + asteroid.radius
            for asteroid in self.asteroids
        )
        self.out_of_fuel = self.ship.fuel <= 0.0
        self.done = (
            self.collision
            or self.out_of_fuel
            or self.steps >= int(self.env_cfg["max_steps"])
        )

        next_distance = self._nearest_mineral_distance()
        if previous_distance is not None and next_distance is not None and mined_count == 0:
            progress = max(0.0, previous_distance - next_distance) / max(float(self.grid_cfg["view_radius"]), 1e-6)
            self.progress_sum += progress

        self.danger_sum += self._asteroid_danger_at_ship()
        if thrust <= float(self.env_cfg["idle_thrust_threshold"]) and mined_count == 0 and not moved:
            self.idle_frames += 1

        return self.observe(), self.done

    def metrics(self):
        fitness_cfg = self.fitness_cfg
        fuel_ratio = self.ship.fuel / float(self.env_cfg["fuel_capacity"])
        fitness = (
            float(fitness_cfg["mineral_reward"]) * self.ship.minerals
            + float(fitness_cfg["progress_reward"]) * self.progress_sum
            + float(fitness_cfg["alive_reward"]) * self.steps
            + float(fitness_cfg["fuel_remaining_reward"]) * fuel_ratio
            - float(fitness_cfg["empty_mine_penalty"]) * self.empty_mine_attempts
            - float(fitness_cfg["idle_penalty"]) * self.idle_frames
            - float(fitness_cfg["danger_penalty"]) * self.danger_sum
        )
        if self.collision:
            fitness -= float(fitness_cfg["collision_penalty"])
        if self.out_of_fuel:
            fitness -= float(fitness_cfg["out_of_fuel_penalty"])
        return EpisodeMetrics(
            fitness=fitness,
            minerals=self.ship.minerals,
            steps=self.steps,
            fuel_remaining=self.ship.fuel,
            collision=self.collision,
            out_of_fuel=self.out_of_fuel,
            empty_mine_attempts=self.empty_mine_attempts,
            idle_frames=self.idle_frames,
            progress=self.progress_sum,
            danger=self.danger_sum,
        )


class HyperNEATController:
    def __init__(self, cppn_network, config):
        self.config = config
        self.substrate_cfg = config["substrate"]
        self.input_nodes, self.hidden_nodes, self.output_nodes = build_substrate(config)
        self.input_hidden_connections = []
        self.hidden_output_connections = []
        self.input_output_connections = []
        self.uses_numpy = bool(self.substrate_cfg["use_numpy"]) and np is not None
        self._decode(cppn_network)

    def _decode(self, cppn_network):
        input_count = len(self.input_nodes)
        hidden_count = len(self.hidden_nodes)
        output_count = len(self.output_nodes)

        if self.uses_numpy:
            self.input_hidden_weights = np.zeros((hidden_count, input_count), dtype=float)
            self.hidden_output_weights = np.zeros((output_count, hidden_count), dtype=float)
            self.input_output_weights = np.zeros((output_count, input_count), dtype=float)
        else:
            self.input_hidden_weights = None
            self.hidden_output_weights = None
            self.input_output_weights = None

        for source_index, source in enumerate(self.input_nodes):
            for target_index, target in enumerate(self.hidden_nodes):
                weight = self._query_weight(cppn_network, source, target)
                if weight != 0.0:
                    if self.uses_numpy:
                        self.input_hidden_weights[target_index, source_index] = weight
                    else:
                        self.input_hidden_connections.append((source_index, target_index, weight))

        for source_index, source in enumerate(self.hidden_nodes):
            for target_index, target in enumerate(self.output_nodes):
                weight = self._query_weight(cppn_network, source, target)
                if weight != 0.0:
                    if self.uses_numpy:
                        self.hidden_output_weights[target_index, source_index] = weight
                    else:
                        self.hidden_output_connections.append((source_index, target_index, weight))

        if bool(self.substrate_cfg["include_direct_input_output"]):
            for source_index, source in enumerate(self.input_nodes):
                for target_index, target in enumerate(self.output_nodes):
                    weight = self._query_weight(cppn_network, source, target)
                    if weight != 0.0:
                        if self.uses_numpy:
                            self.input_output_weights[target_index, source_index] = weight
                        else:
                            self.input_output_connections.append((source_index, target_index, weight))

    def _query_weight(self, cppn_network, source, target):
        distance = math.sqrt(
            (source.x - target.x) ** 2
            + (source.y - target.y) ** 2
            + (source.z - target.z) ** 2
        )
        cppn_input = [
            source.x,
            source.y,
            source.z,
            target.x,
            target.y,
            target.z,
            distance,
            1.0,
        ]
        raw_weight = float(cppn_network.activate(cppn_input)[0])
        if abs(raw_weight) < float(self.substrate_cfg["weight_threshold"]):
            return 0.0
        return raw_weight * float(self.substrate_cfg["weight_scale"])

    def activate(self, inputs):
        if self.uses_numpy:
            input_vector = np.asarray(inputs, dtype=float)
            hidden_raw = self.input_hidden_weights @ input_vector
            hidden = np.tanh(hidden_raw)
            output_raw = self.hidden_output_weights @ hidden
            if bool(self.substrate_cfg["include_direct_input_output"]):
                output_raw = output_raw + self.input_output_weights @ input_vector
            return np.tanh(output_raw).tolist()

        hidden = [0.0] * len(self.hidden_nodes)
        for source_index, target_index, weight in self.input_hidden_connections:
            hidden[target_index] += inputs[source_index] * weight
        hidden = [math.tanh(value) for value in hidden]

        outputs = [0.0] * len(self.output_nodes)
        for source_index, target_index, weight in self.hidden_output_connections:
            outputs[target_index] += hidden[source_index] * weight
        for source_index, target_index, weight in self.input_output_connections:
            outputs[target_index] += inputs[source_index] * weight
        return [math.tanh(value) for value in outputs]

    def connection_counts(self):
        if self.uses_numpy:
            return {
                "input_hidden": int(np.count_nonzero(self.input_hidden_weights)),
                "hidden_output": int(np.count_nonzero(self.hidden_output_weights)),
                "input_output": int(np.count_nonzero(self.input_output_weights)),
            }
        return {
            "input_hidden": len(self.input_hidden_connections),
            "hidden_output": len(self.hidden_output_connections),
            "input_output": len(self.input_output_connections),
        }


def build_substrate(config):
    grid_size = int(config["grid"]["size"])
    hidden_grid_size = int(config["substrate"]["hidden_grid_size"])
    grid_positions = normalized_positions(grid_size)
    hidden_positions = normalized_positions(hidden_grid_size)
    channel_z = {
        "mineral": -0.75,
        "asteroid_danger": -0.25,
        "asteroid_vx": 0.25,
        "asteroid_vy": 0.75,
    }

    input_nodes = []
    for row, y in enumerate(grid_positions):
        for col, x in enumerate(grid_positions):
            for channel in GRID_CHANNELS:
                input_nodes.append(SubstrateNode(x=x, y=y, z=channel_z[channel], name=f"{channel}_{row}_{col}"))

    global_positions = {
        "fuel": (-1.0, 1.0, 1.0),
        "heading_cos": (-0.33, 1.0, 1.0),
        "heading_sin": (0.33, 1.0, 1.0),
        "bias": (1.0, 1.0, 1.0),
    }
    for name in GLOBAL_INPUTS:
        x, y, z = global_positions[name]
        input_nodes.append(SubstrateNode(x=x, y=y, z=z, name=name))

    hidden_nodes = [
        SubstrateNode(x=x, y=y, z=1.25, name=f"hidden_{row}_{col}")
        for row, y in enumerate(hidden_positions)
        for col, x in enumerate(hidden_positions)
    ]

    output_x = [-0.75, -0.25, 0.25, 0.75]
    output_nodes = [
        SubstrateNode(x=x, y=0.0, z=1.75, name=name)
        for x, name in zip(output_x, OUTPUT_NAMES)
    ]
    return input_nodes, hidden_nodes, output_nodes


def outputs_to_action(outputs, config):
    heading_x = float(outputs[0])
    heading_y = float(outputs[1])
    if abs(heading_x) < 1e-6 and abs(heading_y) < 1e-6:
        desired_heading = 0.0
    else:
        desired_heading = math.atan2(heading_y, heading_x)
    thrust = clamp((float(outputs[2]) + 1.0) / 2.0, 0.0, 1.0)
    mine_signal = clamp((float(outputs[3]) + 1.0) / 2.0, 0.0, 1.0)
    return {
        "desired_heading": desired_heading,
        "thrust": thrust,
        "mine": mine_signal > float(config["environment"]["mine_threshold"]),
        "mine_signal": mine_signal,
    }


def run_episode(controller, config, seed):
    env = SpaceMinerEnv(config)
    observation = env.reset(seed)
    done = False
    while not done:
        outputs = controller.activate(observation)
        action = outputs_to_action(outputs, config)
        observation, done = env.step(action)
    return env.metrics()


def summarize_metrics(metrics):
    count = max(len(metrics), 1)
    return {
        "fitness": sum(metric.fitness for metric in metrics) / count,
        "minerals": sum(metric.minerals for metric in metrics) / count,
        "steps": sum(metric.steps for metric in metrics) / count,
        "fuel_remaining": sum(metric.fuel_remaining for metric in metrics) / count,
        "collisions": sum(1 for metric in metrics if metric.collision),
        "out_of_fuel": sum(1 for metric in metrics if metric.out_of_fuel),
        "empty_mine_attempts": sum(metric.empty_mine_attempts for metric in metrics) / count,
        "idle_frames": sum(metric.idle_frames for metric in metrics) / count,
        "progress": sum(metric.progress for metric in metrics) / count,
        "danger": sum(metric.danger for metric in metrics) / count,
        "worst_fitness": min(metric.fitness for metric in metrics),
        "best_fitness": max(metric.fitness for metric in metrics),
    }


def evaluate_controller(controller, config, seeds):
    return [run_episode(controller, config, seed) for seed in seeds]


def eval_genomes_factory(experiment_config, neat_module, neat_config):
    training_seeds = [int(seed) for seed in experiment_config["seeds"]["training"]]

    def eval_genomes(genomes, config):
        del config
        for _, genome in genomes:
            cppn = neat_module.nn.FeedForwardNetwork.create(genome, neat_config)
            controller = HyperNEATController(cppn, experiment_config)
            metrics = evaluate_controller(controller, experiment_config, training_seeds)
            genome.fitness = summarize_metrics(metrics)["fitness"]

    return eval_genomes


def build_neat_config_text(config):
    neat_cfg = config["neat"]
    reset_on_extinction = "1" if bool(neat_cfg["reset_on_extinction"]) else "0"
    enabled_default = "True" if bool(neat_cfg["enabled_default"]) else "False"
    return f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = {float(neat_cfg["fitness_threshold"])}
pop_size              = {int(neat_cfg["pop_size"])}
reset_on_extinction   = {reset_on_extinction}

[DefaultGenome]
num_inputs              = {int(neat_cfg["num_cppn_inputs"])}
num_hidden              = {int(neat_cfg["initial_hidden"])}
num_outputs             = {int(neat_cfg["num_cppn_outputs"])}
initial_connection      = {neat_cfg["initial_connection"]}
feed_forward            = True
compatibility_disjoint_coefficient = {float(neat_cfg["compatibility_disjoint_coefficient"])}
compatibility_weight_coefficient   = {float(neat_cfg["compatibility_weight_coefficient"])}
conn_add_prob           = {float(neat_cfg["conn_add_prob"])}
conn_delete_prob        = {float(neat_cfg["conn_delete_prob"])}
node_add_prob           = {float(neat_cfg["node_add_prob"])}
node_delete_prob        = {float(neat_cfg["node_delete_prob"])}
activation_default      = {neat_cfg["activation_default"]}
activation_options      = {neat_cfg["activation_options"]}
activation_mutate_rate  = {float(neat_cfg["activation_mutate_rate"])}
aggregation_default     = {neat_cfg["aggregation_default"]}
aggregation_options     = {neat_cfg["aggregation_options"]}
aggregation_mutate_rate = {float(neat_cfg["aggregation_mutate_rate"])}
bias_init_mean          = {float(neat_cfg["bias_init_mean"])}
bias_init_stdev         = {float(neat_cfg["bias_init_stdev"])}
bias_replace_rate       = {float(neat_cfg["bias_replace_rate"])}
bias_mutate_rate        = {float(neat_cfg["bias_mutate_rate"])}
bias_mutate_power       = {float(neat_cfg["bias_mutate_power"])}
bias_max_value          = {float(neat_cfg["bias_max_value"])}
bias_min_value          = {float(neat_cfg["bias_min_value"])}
response_init_mean      = {float(neat_cfg["response_init_mean"])}
response_init_stdev     = {float(neat_cfg["response_init_stdev"])}
response_replace_rate   = {float(neat_cfg["response_replace_rate"])}
response_mutate_rate    = {float(neat_cfg["response_mutate_rate"])}
response_mutate_power   = {float(neat_cfg["response_mutate_power"])}
response_max_value      = {float(neat_cfg["response_max_value"])}
response_min_value      = {float(neat_cfg["response_min_value"])}
weight_max_value        = {float(neat_cfg["weight_max_value"])}
weight_min_value        = {float(neat_cfg["weight_min_value"])}
weight_init_mean        = {float(neat_cfg["weight_init_mean"])}
weight_init_stdev       = {float(neat_cfg["weight_init_stdev"])}
weight_mutate_rate      = {float(neat_cfg["weight_mutate_rate"])}
weight_replace_rate     = {float(neat_cfg["weight_replace_rate"])}
weight_mutate_power     = {float(neat_cfg["weight_mutate_power"])}
enabled_default         = {enabled_default}
enabled_mutate_rate     = {float(neat_cfg["enabled_mutate_rate"])}

[DefaultSpeciesSet]
compatibility_threshold = {float(neat_cfg["compatibility_threshold"])}

[DefaultStagnation]
species_fitness_func = {neat_cfg["species_fitness_func"]}
max_stagnation  = {int(neat_cfg["max_stagnation"])}
species_elitism = {int(neat_cfg["species_elitism"])}

[DefaultReproduction]
elitism            = {int(neat_cfg["elitism"])}
survival_threshold = {float(neat_cfg["survival_threshold"])}
""".strip()


def create_neat_config(config, neat_module):
    config_text = build_neat_config_text(config)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".ini", delete=False) as temp_file:
        temp_file.write(config_text)
        temp_path = temp_file.name
    try:
        return neat_module.Config(
            neat_module.DefaultGenome,
            neat_module.DefaultReproduction,
            neat_module.DefaultSpeciesSet,
            neat_module.DefaultStagnation,
            temp_path,
        )
    finally:
        os.unlink(temp_path)


def save_training_outputs(winner, config, winner_controller, training_summary, evaluation_summary):
    winner_path = resolve_output_path(config["outputs"]["winner_path"])
    metrics_path = resolve_output_path(config["outputs"]["metrics_path"])
    winner_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(winner_path, "wb") as winner_file:
        pickle.dump(
            {
                "winner": winner,
                "experiment_config": config,
                "neat_config_text": build_neat_config_text(config),
            },
            winner_file,
        )

    metrics = {
        "training": training_summary,
        "evaluation": evaluation_summary,
        "winner_fitness": winner.fitness,
        "winner_nodes": len(winner.nodes),
        "winner_connections": len(winner.connections),
        "substrate_connections": winner_controller.connection_counts(),
    }
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
        metrics_file.write("\n")

    return winner_path, metrics_path


def train(config, generations_override=None):
    if generations_override is not None:
        config["neat"]["generations"] = int(generations_override)

    neat_module = import_neat()
    neat_config = create_neat_config(config, neat_module)
    population = neat_module.Population(neat_config)
    if bool(config["neat"]["stdout_reporter"]):
        population.add_reporter(neat_module.StdOutReporter(True))
    stats = neat_module.StatisticsReporter()
    population.add_reporter(stats)

    generations = int(config["neat"]["generations"])
    winner = population.run(eval_genomes_factory(config, neat_module, neat_config), generations)

    winner_cppn = neat_module.nn.FeedForwardNetwork.create(winner, neat_config)
    winner_controller = HyperNEATController(winner_cppn, config)
    training_metrics = evaluate_controller(winner_controller, config, config["seeds"]["training"])
    evaluation_metrics = evaluate_controller(winner_controller, config, config["seeds"]["evaluation"])
    training_summary = summarize_metrics(training_metrics)
    evaluation_summary = summarize_metrics(evaluation_metrics)
    winner_path, metrics_path = save_training_outputs(
        winner,
        config,
        winner_controller,
        training_summary,
        evaluation_summary,
    )

    print("\nTraining complete")
    print(f"Winner fitness: {winner.fitness:.3f}")
    print(f"Winner nodes: {len(winner.nodes)}")
    print(f"Winner CPPN connections: {len(winner.connections)}")
    print(f"Decoded substrate connections: {winner_controller.connection_counts()}")
    print(f"Training seeds summary: {training_summary}")
    print(f"Evaluation seeds summary: {evaluation_summary}")
    print(f"Saved winner to {winner_path}")
    print(f"Saved metrics to {metrics_path}")
    return winner_controller, winner, neat_config


def render_controller(controller, config, seed):
    import pygame

    pygame.init()
    width = int(config["environment"]["width"])
    height = int(config["environment"]["height"])
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("HyperNEAT v1 - Space Miner")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    env = SpaceMinerEnv(config)
    observation = env.reset(seed)
    done = False

    colors = {
        "black": (0, 0, 0),
        "white": (240, 240, 240),
        "ship": (70, 130, 255),
        "mineral": (255, 220, 70),
        "asteroid": (230, 70, 70),
        "grid": (45, 55, 65),
    }

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        outputs = controller.activate(observation)
        action = outputs_to_action(outputs, config)
        observation, done = env.step(action)

        screen.fill(colors["black"])
        if bool(config["render"]["show_grid"]):
            draw_local_grid(screen, env, config, colors["grid"])
        for mineral in env.minerals:
            pygame.draw.circle(screen, colors["mineral"], (int(mineral.x), int(mineral.y)), int(mineral.radius))
        for asteroid in env.asteroids:
            pygame.draw.circle(screen, colors["asteroid"], (int(asteroid.x), int(asteroid.y)), int(asteroid.radius))
        draw_ship(screen, env.ship, colors)
        draw_stats(screen, font, env, action, colors["white"])
        pygame.display.flip()
        clock.tick(int(config["render"]["fps"]))

    pygame.time.wait(1000)
    pygame.quit()


def draw_local_grid(screen, env, config, color):
    import pygame

    grid_size = int(config["grid"]["size"])
    view_radius = float(config["grid"]["view_radius"])
    cell_size = (2.0 * view_radius) / grid_size
    left = env.ship.x - view_radius
    top = env.ship.y - view_radius
    for index in range(grid_size + 1):
        offset = index * cell_size
        pygame.draw.line(screen, color, (left + offset, top), (left + offset, top + 2 * view_radius), 1)
        pygame.draw.line(screen, color, (left, top + offset), (left + 2 * view_radius, top + offset), 1)


def draw_ship(screen, ship, colors):
    import pygame

    pygame.draw.circle(screen, colors["ship"], (int(ship.x), int(ship.y)), int(ship.radius))
    points = [
        (
            ship.x + ship.radius * math.cos(ship.angle),
            ship.y + ship.radius * math.sin(ship.angle),
        ),
        (
            ship.x + ship.radius * math.cos(ship.angle + 2.5),
            ship.y + ship.radius * math.sin(ship.angle + 2.5),
        ),
        (
            ship.x + ship.radius * math.cos(ship.angle - 2.5),
            ship.y + ship.radius * math.sin(ship.angle - 2.5),
        ),
    ]
    pygame.draw.polygon(screen, colors["white"], points)


def draw_stats(screen, font, env, action, color):
    stats = [
        f"Step: {env.steps}",
        f"Minerals: {env.ship.minerals}",
        f"Fuel: {env.ship.fuel:.1f}",
        f"Thrust: {action['thrust']:.2f}",
        f"Mine: {action['mine_signal']:.2f}",
    ]
    for index, stat in enumerate(stats):
        text = font.render(stat, True, color)
        screen.blit(text, (10, 10 + index * 28))


def parse_args():
    parser = argparse.ArgumentParser(description="HyperNEAT v1 Space Miner experiment")
    parser.add_argument("--config", help="Path to a JSON experiment config")
    parser.add_argument("--write-default-config", help="Write the built-in default config JSON to this path and exit")
    parser.add_argument("--generations", type=int, help="Override neat.generations")
    parser.add_argument("--render-winner", action="store_true", help="Render the winner after training")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering even if config enables it")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.write_default_config:
        write_default_config(args.write_default_config)
        return

    config = load_experiment_config(args.config)
    if args.no_render:
        config["render"]["winner"] = False
    if args.render_winner:
        config["render"]["winner"] = True

    controller, _, _ = train(config, generations_override=args.generations)
    if bool(config["render"]["winner"]):
        render_controller(controller, config, int(config["render"]["seed"]))


if __name__ == "__main__":
    main()
