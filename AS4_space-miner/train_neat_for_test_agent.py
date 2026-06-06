import argparse
import math
import os
import pickle
import shutil
from datetime import datetime
import neat
import pygame

# Training is headless. This must be set before importing miner_harness because
# that module creates a pygame display at import time.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")



from miner_harness import Asteroid, Mineral, Spaceship


WIDTH, HEIGHT = 800, 600
MAX_DISTANCE = math.hypot(WIDTH, HEIGHT)
MAX_FRAMES = 5000
ASTEROID_DANGER_MARGIN = 20
MINERAL_APPROACH_WEIGHT = 0.02
MINERAL_PROGRESS_WEIGHT = 0.05
FUEL_EFFICIENCY_WEIGHT = 25.0


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
    asteroid_velocity_x = closest_asteroid.speed_x / ship.speed
    asteroid_velocity_y = closest_asteroid.speed_y / ship.speed

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
        asteroid_velocity_x,
        asteroid_velocity_y,
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
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(8)]

    alive_time = 0
    death_reason = "time_limit"
    ship_velocity_x = 0
    ship_velocity_y = 0
    mineral_progress = 0
    mineral_approach = 0
    mineral_best_distances = {}
    fuel_used = 0
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
                    fuel_used,
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
        if closest_mineral and closest_mineral not in mineral_best_distances:
            mineral_best_distances[closest_mineral] = target_distance_before

        output = net.activate(inputs)

        # Match test_agent.py exactly so the saved genome behaves the same
        # during final playback.
        ship.angle += (output[0] * 2 - 1) * 0.1
        if output[1] > 0.5:
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
            fuel_before_move = ship.fuel
            ship.move(dx, dy)
            fuel_used += max(0, fuel_before_move - ship.fuel)
            ship_velocity_x = dx
            ship_velocity_y = dy
        else:
            ship_velocity_x = 0
            ship_velocity_y = 0
            idle_time += 1

        if closest_mineral and target_distance_before is not None:
            target_distance_after = distance_between(ship, closest_mineral)
            mineral_distance_delta = target_distance_before - target_distance_after
            if mineral_distance_delta > 0:
                mineral_approach += mineral_distance_delta

            best_distance = mineral_best_distances[closest_mineral]
            if target_distance_after < best_distance:
                mineral_progress += best_distance - target_distance_after
                mineral_best_distances[closest_mineral] = target_distance_after

        if output[2] > 0.5:
            mine_attempts += 1
            old_mineral_count = ship.minerals
            old_fuel = ship.fuel
            ship.mine(minerals)
            if ship.minerals > old_mineral_count:
                successful_mines += 1
                if ship.fuel == old_fuel and ship.fuel < 100.0:
                    ship.fuel = min(100.0, ship.fuel + 10.0)
            if len(minerals) < 3:
                minerals.extend(Mineral() for _ in range(2))
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

        asteroid_collision = (
            distance_between(ship, closest_asteroid)
            < ship.radius + closest_asteroid.radius
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
        fuel_used,
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
    fuel_used,
    asteroid_danger,
    idle_time,
    mine_attempts,
    successful_mines,
):
    test_score = (alive_time / 4) + (ship.minerals * 100)
    fuel_efficiency = ship.minerals / max(fuel_used, 1)
    wasted_mines = max(0, mine_attempts - successful_mines)

    fitness = test_score
    fitness += mineral_progress * MINERAL_PROGRESS_WEIGHT
    fitness += mineral_approach * MINERAL_APPROACH_WEIGHT
    fitness += fuel_efficiency * FUEL_EFFICIENCY_WEIGHT
    fitness += ship.fuel * 0.2
    fitness -= asteroid_danger * 3.0
    fitness -= idle_time * 0.02
    fitness -= wasted_mines * 0.03

    if death_reason == "asteroid_collision":
        fitness -= 500
    elif death_reason == "out_of_fuel":
        fitness -= 150

    return fitness, {
        "fitness": fitness,
        "test_score": test_score,
        "alive_time": alive_time,
        "minerals": ship.minerals,
        "fuel": ship.fuel,
        "fuel_used": fuel_used,
        "fuel_efficiency": fuel_efficiency,
        "mineral_approach": mineral_approach,
        "mineral_progress": mineral_progress,
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


def backup_existing_file(path):
    if not os.path.exists(path):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.{timestamp}.bak"
    shutil.copy2(path, backup_path)
    return backup_path


def train(config_path, output_path, generations):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    if config.genome_config.num_inputs != 13:
        raise ValueError("test_agent.py expects a genome configured with 13 inputs")
    if config.genome_config.num_outputs != 3:
        raise ValueError("test_agent.py expects a genome configured with 3 outputs")

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(eval_genomes, generations)

    backup_path = backup_existing_file(output_path)
    with open(output_path, "wb") as f:
        pickle.dump(winner, f)

    if backup_path:
        print(f"Backed up previous winner to: {backup_path}")
    print(f"Saved winner genome to: {output_path}")
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
        default=10,
        help="Number of NEAT generations to run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config, args.output, args.generations)
