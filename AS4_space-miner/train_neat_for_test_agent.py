import argparse
import math
import os
import pickle
import neat
import pygame

# Training is headless. This must be set before importing miner_harness because
# that module creates a pygame display at import time.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from miner_harness import Asteroid, Mineral, Spaceship
from train_neat_for_test_agent_config import (
    ASTEROID_COLLISION_PENALTY,
    ASTEROID_DANGER_MARGIN,
    ASTEROID_DANGER_WEIGHT,
    DEFAULT_GENERATIONS,
    EXPECTED_INPUTS,
    EXPECTED_OUTPUTS,
    FUEL_EFFICIENCY_WEIGHT,
    HEIGHT,
    IDLE_PENALTY_WEIGHT,
    INITIAL_ASTEROID_COUNT,
    INITIAL_MINERAL_COUNT,
    MAX_DISTANCE,
    MAX_FRAMES,
    MINE_THRESHOLD,
    MINERAL_APPROACH_WEIGHT,
    MINERAL_ALIGNMENT_WEIGHT,
    MINERAL_PROGRESS_WEIGHT,
    MINERAL_VELOCITY_WEIGHT,
    MINERAL_REFILL_COUNT,
    MINERAL_REFILL_THRESHOLD,
    MINING_REFUEL_AMOUNT,
    OUT_OF_FUEL_PENALTY,
    REMAINING_FUEL_WEIGHT,
    THRUST_THRESHOLD,
    WASTED_MINES_WEIGHT,
    TURN_RATE,
    WIDTH,
)


def relative_position(source, target, width=WIDTH, height=HEIGHT):
    dx = target.x - source.x
    dy = target.y - source.y
    half_width = width / 2
    half_height = height / 2

    if dx > half_width:
        dx -= width
    elif dx < -half_width:
        dx += width

    if dy > half_height:
        dy -= height
    elif dy < -half_height:
        dy += height

    return dx, dy


def distance_between(source, target, width=WIDTH, height=HEIGHT):
    dx, dy = relative_position(source, target, width, height)
    return math.hypot(dx, dy)


def relative_angle_to(ship, target):
    dx, dy = relative_position(ship, target)
    target_angle = math.atan2(dy, dx)
    angle_delta = target_angle - ship.angle
    return math.atan2(math.sin(angle_delta), math.cos(angle_delta))


def normalized_relative_vector(source, target):
    dx, dy = relative_position(source, target)
    return dx / (WIDTH / 2), dy / (HEIGHT / 2)


def asteroid_in_front(ship, asteroid):
    dx, dy = relative_position(ship, asteroid)
    distance = math.hypot(dx, dy)
    if distance == 0:
        return 1

    heading_x = math.cos(ship.angle)
    heading_y = math.sin(ship.angle)
    return max(0, (heading_x * dx + heading_y * dy) / distance)


def asteroid_time_to_collision_signal(
    ship,
    asteroid,
    ship_velocity_x,
    ship_velocity_y,
    horizon=120,
):
    dx, dy = relative_position(ship, asteroid)
    relative_vx = asteroid.speed_x - ship_velocity_x
    relative_vy = asteroid.speed_y - ship_velocity_y
    relative_speed_sq = relative_vx * relative_vx + relative_vy * relative_vy
    if relative_speed_sq == 0:
        return 0

    time_to_closest = -(dx * relative_vx + dy * relative_vy) / relative_speed_sq
    if time_to_closest < 0 or time_to_closest > horizon:
        return 0

    closest_x = dx + relative_vx * time_to_closest
    closest_y = dy + relative_vy * time_to_closest
    closest_distance = math.hypot(closest_x, closest_y)
    danger_radius = ship.radius + asteroid.radius + ASTEROID_DANGER_MARGIN
    if closest_distance > danger_radius:
        return 0

    return 1 - (time_to_closest / horizon)


def observe(ship, minerals, asteroids, ship_velocity_x, ship_velocity_y):
    closest_mineral = min(
        (m for m in minerals),
        key=lambda m: distance_between(ship, m),
        default=None,
    )
    closest_asteroid = min(asteroids, key=lambda a: distance_between(ship, a))

    mineral_distance = (
        distance_between(ship, closest_mineral) / MAX_DISTANCE
        if closest_mineral
        else 0
    )
    mineral_relative_angle = (
        relative_angle_to(ship, closest_mineral) / math.pi
        if closest_mineral
        else 0
    )
    asteroid_distance = distance_between(ship, closest_asteroid) / MAX_DISTANCE
    asteroid_relative_angle = relative_angle_to(ship, closest_asteroid) / math.pi
    mineral_relative_x, mineral_relative_y = (
        normalized_relative_vector(ship, closest_mineral)
        if closest_mineral
        else (0, 0)
    )
    asteroid_relative_x, asteroid_relative_y = normalized_relative_vector(
        ship,
        closest_asteroid,
    )
    asteroid_relative_velocity_x = (
        closest_asteroid.speed_x - ship_velocity_x
    ) / ship.speed
    asteroid_relative_velocity_y = (
        closest_asteroid.speed_y - ship_velocity_y
    ) / ship.speed

    inputs = [
        mineral_distance,
        mineral_relative_angle,
        asteroid_distance,
        asteroid_relative_angle,
        ship.fuel / 100.0,
        mineral_relative_x,
        mineral_relative_y,
        asteroid_relative_x,
        asteroid_relative_y,
        asteroid_relative_velocity_x,
        asteroid_relative_velocity_y,
        asteroid_in_front(ship, closest_asteroid),
        asteroid_time_to_collision_signal(
            ship,
            closest_asteroid,
            ship_velocity_x,
            ship_velocity_y,
        ),
    ]
    return inputs, closest_mineral, closest_asteroid


def reset_fixed_harness_sequence():
    Mineral._index = 0
    Asteroid._index = 0


def run_episode(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    reset_fixed_harness_sequence()

    ship = Spaceship()
    minerals = [Mineral() for _ in range(INITIAL_MINERAL_COUNT)]
    asteroids = [Asteroid() for _ in range(INITIAL_ASTEROID_COUNT)]

    alive_time = 0
    death_reason = "time_limit"
    ship_velocity_x = 0
    ship_velocity_y = 0
    mineral_progress = 0
    mineral_approach = 0
    mineral_alignment = 0
    mineral_velocity = 0
    mineral_best_distances = {}
    fuel_used = 0
    asteroid_avoidance_fuel = 0
    asteroid_danger = 0
    idle_time = 0
    mine_attempts = 0
    successful_mines = 0

    while alive_time <= MAX_FRAMES:
        alive_time += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                death_reason = "window_closed"
                return score_episode(
                    ship,
                    alive_time,
                    death_reason,
                    mineral_progress,
                    mineral_approach,
                    mineral_alignment,
                    mineral_velocity,
                    fuel_used,
                    asteroid_avoidance_fuel,
                    asteroid_danger,
                    idle_time,
                    mine_attempts,
                    successful_mines,
                )

        inputs, closest_mineral, closest_asteroid = observe(
            ship,
            minerals,
            asteroids,
            ship_velocity_x,
            ship_velocity_y,
        )
        target_distance_before = (
            distance_between(ship, closest_mineral) if closest_mineral else None
        )
        asteroid_threat_before = False
        if closest_asteroid:
            asteroid_clearance_before = (
                distance_between(ship, closest_asteroid)
                - ship.radius
                - closest_asteroid.radius
            )
            asteroid_threat_before = asteroid_clearance_before < ASTEROID_DANGER_MARGIN

        if closest_mineral and closest_mineral not in mineral_best_distances:
            mineral_best_distances[closest_mineral] = target_distance_before

        output = net.activate(inputs)

        # Match test_agent.py exactly so the saved genome behaves the same
        # during final playback.
        ship.angle += (output[0] * 2 - 1) * TURN_RATE
        if output[1] > THRUST_THRESHOLD:
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
            fuel_before_move = ship.fuel
            ship.move(dx, dy)
            fuel_delta = max(0, fuel_before_move - ship.fuel)
            fuel_used += fuel_delta
            if asteroid_threat_before:
                asteroid_avoidance_fuel += fuel_delta
            ship_velocity_x = dx
            ship_velocity_y = dy
        else:
            ship_velocity_x = 0
            ship_velocity_y = 0
            idle_time += 1

        if closest_mineral and target_distance_before is not None:
            target_distance_after = distance_between(ship, closest_mineral)
            mineral_distance_delta = target_distance_before - target_distance_after
            mineral_alignment += max(0, math.cos(relative_angle_to(ship, closest_mineral)))

            mineral_dx, mineral_dy = relative_position(ship, closest_mineral)
            mineral_distance = max(math.hypot(mineral_dx, mineral_dy), 1)
            movement_toward_mineral = (
                ship_velocity_x * mineral_dx + ship_velocity_y * mineral_dy
            ) / mineral_distance
            if movement_toward_mineral > 0:
                mineral_velocity += movement_toward_mineral

            if mineral_distance_delta > 0:
                mineral_approach += mineral_distance_delta

            best_distance = mineral_best_distances[closest_mineral]
            if target_distance_after < best_distance:
                mineral_progress += best_distance - target_distance_after
                mineral_best_distances[closest_mineral] = target_distance_after

        if output[2] > MINE_THRESHOLD:
            mine_attempts += 1
            old_mineral_count = ship.minerals
            old_fuel = ship.fuel
            ship.mine(minerals)
            if ship.minerals > old_mineral_count:
                successful_mines += 1
                if ship.fuel == old_fuel and ship.fuel < 100.0:
                    ship.fuel = min(100.0, ship.fuel + MINING_REFUEL_AMOUNT)
            if len(minerals) < MINERAL_REFILL_THRESHOLD:
                minerals.extend(Mineral() for _ in range(MINERAL_REFILL_COUNT))
            mineral_best_distances = {
                mineral: distance
                for mineral, distance in mineral_best_distances.items()
                if mineral in minerals
            }

        for asteroid in asteroids:
            asteroid.move()

        asteroid_clearance = min(
            distance_between(ship, asteroid) - ship.radius - asteroid.radius
            for asteroid in asteroids
        )
        if asteroid_clearance < ASTEROID_DANGER_MARGIN:
            asteroid_danger += (
                ASTEROID_DANGER_MARGIN - max(0, asteroid_clearance)
            ) / ASTEROID_DANGER_MARGIN

        asteroid_collision = any(
            distance_between(ship, asteroid) < ship.radius + asteroid.radius
            for asteroid in asteroids
        )
        out_of_fuel = ship.fuel <= 0

        if asteroid_collision:
            death_reason = "asteroid_collision"
            break
        if out_of_fuel:
            death_reason = "out_of_fuel"
            break

    return score_episode(
        ship,
        alive_time,
        death_reason,
        mineral_progress,
        mineral_approach,
        mineral_alignment,
        mineral_velocity,
        fuel_used,
        asteroid_avoidance_fuel,
        asteroid_danger,
        idle_time,
        mine_attempts,
        successful_mines,
    )


def score_episode(
    ship,
    alive_time,
    death_reason,
    mineral_progress,
    mineral_approach,
    mineral_alignment,
    mineral_velocity,
    fuel_used,
    asteroid_avoidance_fuel,
    asteroid_danger,
    idle_time,
    mine_attempts,
    successful_mines,
):
    test_score = (alive_time / 4) + (ship.minerals * 100)
    effective_fuel_used = fuel_used + asteroid_avoidance_fuel
    fuel_efficiency = ship.minerals / max(effective_fuel_used, 1)
    wasted_mines = max(0, mine_attempts - successful_mines)

    fitness = test_score
    fitness += mineral_progress * MINERAL_PROGRESS_WEIGHT
    fitness += mineral_approach * MINERAL_APPROACH_WEIGHT
    fitness += mineral_alignment * MINERAL_ALIGNMENT_WEIGHT
    fitness += mineral_velocity * MINERAL_VELOCITY_WEIGHT
    fitness += fuel_efficiency * FUEL_EFFICIENCY_WEIGHT
    fitness += ship.fuel * REMAINING_FUEL_WEIGHT
    fitness -= asteroid_danger * ASTEROID_DANGER_WEIGHT
    fitness -= idle_time * IDLE_PENALTY_WEIGHT
    fitness -= wasted_mines * WASTED_MINES_WEIGHT

    if death_reason == "asteroid_collision":
        fitness -= ASTEROID_COLLISION_PENALTY
    elif death_reason == "out_of_fuel":
        fitness -= OUT_OF_FUEL_PENALTY

    return fitness, {
        "fitness": fitness,
        "test_score": test_score,
        "alive_time": alive_time,
        "minerals": ship.minerals,
        "fuel": ship.fuel,
        "fuel_used": fuel_used,
        "asteroid_avoidance_fuel": asteroid_avoidance_fuel,
        "effective_fuel_used": effective_fuel_used,
        "fuel_efficiency": fuel_efficiency,
        "mineral_approach": mineral_approach,
        "mineral_progress": mineral_progress,
        "mineral_alignment": mineral_alignment,
        "mineral_velocity": mineral_velocity,
        "death_reason": death_reason,
    }


def eval_genomes(genomes, config):
    best = None
    for genome_id, genome in genomes:
        fitness, metrics = run_episode(genome, config)
        genome.fitness = fitness
        genome.metrics = metrics

        if best is None or genome.fitness > best.fitness:
            best = genome

    print(
        "Generation best: "
        f"fitness={best.metrics['fitness']:.2f} "
        f"score={best.metrics['test_score']:.2f} "
        f"minerals={best.metrics['minerals']} "
        f"approach={best.metrics['mineral_approach']:.1f} "
        f"fuel_eff={best.metrics['fuel_efficiency']:.3f} "
        f"alive={best.metrics['alive_time']} "
        f"death={best.metrics['death_reason']}"
    )


def train(config_path, output_path, generations):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    if config.genome_config.num_inputs != EXPECTED_INPUTS:
        raise ValueError(
            f"test_agent.py expects a genome configured with {EXPECTED_INPUTS} inputs"
        )
    if config.genome_config.num_outputs != EXPECTED_OUTPUTS:
        raise ValueError(
            f"test_agent.py expects a genome configured with {EXPECTED_OUTPUTS} outputs"
        )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes, generations)

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"Saved winner to: {output_path}")

    return winner


def parse_args():
    local_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(
        description=(
            "Train a NEAT genome using the same inputs and action conventions "
            "as test_agent.py."
        )
    )
    parser.add_argument(
        "--config",
        default=os.path.join(local_dir, "neat_config.txt"),
        help="Path to the neat-python config file.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(local_dir, "winner.pkl"),
        help="Path for the saved genome pickle. test_agent.py loads winner.pkl by default.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_GENERATIONS,
        help="Number of NEAT generations to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.output, args.generations)
