import pickle

import pygame
import random
import math
import os
import neat
import time
import configparser

# Initialize pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("NEAT - Space Miner Training")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

GENERATIONS = 10
SENSOR_TARGET_COUNT = 2
FAR_FROM_MINERAL_DISTANCE = 0.35

DEFAULT_FITNESS_WEIGHTS = {
    "minerals": 25.0,
    "alive_time": 0.01,
    "mineral_progress": 0.01,
    "mineral_approach": 0.05,
    "mineral_retreat_penalty": 0.025,
    "mineral_heading_alignment": 0.005,
    "idle_penalty": 0.0035,
    "indecision_penalty": 0.01,
    "fuel_efficiency": 0.05,
    "asteroid_collision_penalty": 0.001,
    "steering_penalty": 0.001,
    "asteroid_proximity_penalty": 0.01
}

ASTEROID_PROXIMITY_THRESHOLD = 20


def wrap_delta(delta, size):
    half = size / 2
    if delta > half:
        delta -= size
    elif delta < -half:
        delta += size
    return delta


def relative_position(source, target):
    return (
        wrap_delta(target.x - source.x, WIDTH),
        wrap_delta(target.y - source.y, HEIGHT),
    )


def distance_between(source, target):
    dx, dy = relative_position(source, target)
    return math.hypot(dx, dy)


def load_fitness_weights(config_file):
    parser = configparser.ConfigParser()
    parser.read(config_file)

    weights = DEFAULT_FITNESS_WEIGHTS.copy()
    if parser.has_section("FitnessWeights"):
        for weight_name in weights:
            if parser.has_option("FitnessWeights", weight_name):
                weights[weight_name] = parser.getfloat("FitnessWeights", weight_name)

    return weights

# Game Classes (same as before)
class Spaceship:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 5
        self.angle = 0
        self.fuel = 100
        self.fuel_used = 0
        self.distance_travelled = 0
        self.minerals = 0
        self.radius = 15
        self.velocity_x = 0
        self.velocity_y = 0

    def move(self, dx, dy):
        if self.fuel > 0:
            self.x = (self.x + dx) % WIDTH
            self.y = (self.y + dy) % HEIGHT
            self.velocity_x = dx
            self.velocity_y = dy
            movement_distance = math.hypot(dx, dy)
            fuel_cost = min(0.1, self.fuel)
            self.fuel -= fuel_cost
            self.fuel_used += fuel_cost
            self.distance_travelled += movement_distance
            return movement_distance
        self.velocity_x = 0
        self.velocity_y = 0
        return 0

    def mine(self, minerals):
        for mineral in minerals[:]:
            dist = distance_between(self, mineral)
            if dist < self.radius + mineral.radius:
                minerals.remove(mineral)
                self.minerals += 1
                self.fuel = min(100, self.fuel + 10)

    def draw(self):
        pygame.draw.circle(screen, BLUE, (int(self.x), int(self.y)), self.radius)
        points = [
            (self.x + self.radius * math.cos(self.angle), 
            self.y + self.radius * math.sin(self.angle)),
            (self.x + self.radius * math.cos(self.angle + 2.5), 
            self.y + self.radius * math.sin(self.angle + 2.5)),
            (self.x + self.radius * math.cos(self.angle - 2.5), 
            self.y + self.radius * math.sin(self.angle - 2.5))
        ]
        pygame.draw.polygon(screen, WHITE, points)

class Mineral:
    def __init__(self):
        self.x = random.randint(20, WIDTH - 20)
        self.y = random.randint(20, HEIGHT - 20)
        self.radius = 10

    def draw(self):
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), self.radius)

class Asteroid:
    _coordinates = [(553,323), (124,303), (556,82), (490,425), (120,218), (774,142), (240,308), (64,475), (651,227), (462,77),]
    _radius = [28, 15, 15, 22, 29, 17, 24, 22, 23, 17,]
    _speed = [(1.78425438642474,0.6732917959108602), (0.3526239655425698,-1.295737594576594), (0.9002543984134839,-0.1395601523204939), 
              (-0.5781368926686516,1.349036516126493), (1.8150210404495417,1.2741322355662592), (1.2831373928635355,0.8453257857927245), 
              (-1.8228539352311386,-1.38841117002117), (-0.9822217473337984,1.8758459317074854), (0.04980714264573827,0.5986537138756898), 
              (0.8160687662014605,-1.0618283414551777)]
    _index = 0

    @staticmethod
    def get_next_coord():
        coord = Asteroid._coordinates[Asteroid._index]
        radius = Asteroid._radius[Asteroid._index]
        speed = Asteroid._speed[Asteroid._index]
        Asteroid._index = (Asteroid._index + 1) % len(Asteroid._coordinates)  # Cycle through
        return coord, radius, speed
    
    def __init__(self):
        (self.x, self.y), self.radius, (self.speed_x, self.speed_y) = Asteroid.get_next_coord()  # Get next fixed coordinate

    def move(self):
        self.x = (self.x + self.speed_x) % WIDTH
        self.y = (self.y + self.speed_y) % HEIGHT

    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)

def calculate_fitness(
    ship,
    alive_time,
    mineral_progress,
    mineral_approach,
    mineral_heading_alignment,
    idle_time,
    asteroid_collision,
    cumulative_steering,
    asteroid_proximity,
    fitness_weights,
    mineral_retreat=0,
    indecision=0,
    return_components=False
):
    fuel_efficiency = ship.minerals / max(ship.fuel_used, 1)
    asteroid_penalty = alive_time if asteroid_collision else 0
    get_weight = lambda name: fitness_weights.get(name, DEFAULT_FITNESS_WEIGHTS[name])

    components = {
        "minerals": ship.minerals * get_weight("minerals"),
        "alive": alive_time * get_weight("alive_time"),
        "progress": mineral_progress * get_weight("mineral_progress"),
        "approach": mineral_approach * get_weight("mineral_approach"),
        "heading": mineral_heading_alignment * get_weight("mineral_heading_alignment"),
        "fuel_efficiency": fuel_efficiency * get_weight("fuel_efficiency"),
        "idle": -idle_time * get_weight("idle_penalty"),
        "retreat": -mineral_retreat * get_weight("mineral_retreat_penalty"),
        "indecision": -indecision * get_weight("indecision_penalty"),
        "collision": -asteroid_penalty * get_weight("asteroid_collision_penalty"),
        "steering": -cumulative_steering * get_weight("steering_penalty"),
        "asteroid_proximity": -asteroid_proximity * get_weight("asteroid_proximity_penalty"),
    }

    fitness = sum(components.values())
    if return_components:
        return fitness, components
    return fitness

def relative_angle_to(ship, target):
    dx, dy = relative_position(ship, target)
    target_angle = math.atan2(dy, dx)
    angle_delta = target_angle - ship.angle
    return math.atan2(math.sin(angle_delta), math.cos(angle_delta))


def normalized_relative_vector(source, target):
    dx, dy = relative_position(source, target)
    return dx / (WIDTH / 2), dy / (HEIGHT / 2)


def nearest_objects(ship, objects, count):
    return sorted(objects, key=lambda obj: distance_between(ship, obj))[:count]


def mineral_sensor_values(ship, mineral, max_distance):
    if mineral is None:
        return [0, 0, 0, 0]

    relative_x, relative_y = normalized_relative_vector(ship, mineral)
    return [
        distance_between(ship, mineral) / max_distance,
        relative_angle_to(ship, mineral) / math.pi,
        relative_x,
        relative_y,
    ]


def asteroid_sensor_values(ship, asteroid, max_distance):
    if asteroid is None:
        return [0, 0, 0, 0, 0, 0, 0, 0]

    relative_x, relative_y = normalized_relative_vector(ship, asteroid)
    return [
        distance_between(ship, asteroid) / max_distance,
        relative_angle_to(ship, asteroid) / math.pi,
        relative_x,
        relative_y,
        asteroid.speed_x / ship.speed,
        asteroid.speed_y / ship.speed,
        asteroid_in_front(ship, asteroid),
        asteroid_time_to_collision_signal(ship, asteroid),
    ]


def build_neat_inputs(ship, minerals, asteroids, max_distance):
    inputs = []

    nearest_minerals = nearest_objects(ship, minerals, SENSOR_TARGET_COUNT)
    for index in range(SENSOR_TARGET_COUNT):
        mineral = nearest_minerals[index] if index < len(nearest_minerals) else None
        inputs.extend(mineral_sensor_values(ship, mineral, max_distance))

    nearest_asteroids = nearest_objects(ship, asteroids, SENSOR_TARGET_COUNT)
    for index in range(SENSOR_TARGET_COUNT):
        asteroid = nearest_asteroids[index] if index < len(nearest_asteroids) else None
        inputs.extend(asteroid_sensor_values(ship, asteroid, max_distance))

    inputs.append(ship.fuel / 100.0)
    return inputs


def asteroid_in_front(ship, asteroid):
    dx, dy = relative_position(ship, asteroid)
    distance = math.hypot(dx, dy)
    if distance == 0:
        return 1

    heading_x = math.cos(ship.angle)
    heading_y = math.sin(ship.angle)
    return max(0, (heading_x * dx + heading_y * dy) / distance)


def asteroid_time_to_collision_signal(ship, asteroid, horizon=120):
    dx, dy = relative_position(ship, asteroid)
    relative_vx = asteroid.speed_x - ship.velocity_x
    relative_vy = asteroid.speed_y - ship.velocity_y
    relative_speed_sq = relative_vx * relative_vx + relative_vy * relative_vy
    if relative_speed_sq == 0:
        return 0

    time_to_closest = -(
        dx * relative_vx + dy * relative_vy
    ) / relative_speed_sq
    if time_to_closest < 0 or time_to_closest > horizon:
        return 0

    closest_x = dx + relative_vx * time_to_closest
    closest_y = dy + relative_vy * time_to_closest
    closest_distance = math.hypot(closest_x, closest_y)
    danger_radius = ship.radius + asteroid.radius + ASTEROID_PROXIMITY_THRESHOLD
    if closest_distance > danger_radius:
        return 0

    return 1 - (time_to_closest / horizon)


def run_simulation(genome, config, visualizer=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness_weights = getattr(config, "fitness_weights", DEFAULT_FITNESS_WEIGHTS)
    ship = Spaceship()
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(8)]
    alive_time = 0
    mineral_progress = 0
    mineral_approach = 0
    mineral_retreat = 0
    mineral_heading_alignment = 0
    mineral_best_distances = {}
    idle_time = 0
    indecision = 0
    frames_without_mineral_progress = 0
    cumulative_steering = 0
    asteroid_proximity = 0
    max_distance = math.hypot(WIDTH, HEIGHT)
    
    while True:
        alive_time += 1
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Find closest objects
        closest_mineral = min((m for m in minerals), 
                            key=lambda m: distance_between(ship, m),
                            default=None)
        if closest_mineral and closest_mineral not in mineral_best_distances:
            mineral_best_distances[closest_mineral] = distance_between(ship, closest_mineral)
        closest_asteroid = min((a for a in asteroids), 
                              key=lambda a: distance_between(ship, a))
        target_distance_before = (
            distance_between(ship, closest_mineral)
            if closest_mineral else None
        )

        # Get inputs (handle case where all minerals are collected)
        inputs = build_neat_inputs(ship, minerals, asteroids, max_distance)
        
        # Get actions from network
        output = net.activate(inputs)
        
        # Execute actions
        movement_distance = 0
        steering_command = (output[0] * 2 - 1) * 0.1  # Turn (-1 to 1)
        ship.angle += steering_command
        cumulative_steering += abs(steering_command)  # Track total steering magnitude
        if output[1] > 0.5:  # Thrust
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
            movement_distance = ship.move(dx, dy)
        else:
            ship.velocity_x = 0
            ship.velocity_y = 0
        if movement_distance == 0:
            idle_time += 1
            indecision += 1
            if abs(steering_command) > 0.03:
                indecision += 0.5
            if (
                closest_mineral
                and target_distance_before is not None
                and target_distance_before / max_distance > FAR_FROM_MINERAL_DISTANCE
            ):
                indecision += 0.5
        else:
            asteroid_clearance = max(
                0,
                distance_between(ship, closest_asteroid) - ship.radius - closest_asteroid.radius
            )
            if asteroid_clearance < ASTEROID_PROXIMITY_THRESHOLD:
                asteroid_proximity += (
                    ASTEROID_PROXIMITY_THRESHOLD - asteroid_clearance
                ) / ASTEROID_PROXIMITY_THRESHOLD
        if closest_mineral:
            target_distance_after = distance_between(ship, closest_mineral)
            mineral_distance_delta = target_distance_before - target_distance_after
            if mineral_distance_delta > 0:
                mineral_approach += mineral_distance_delta
                frames_without_mineral_progress = 0
            elif mineral_distance_delta < 0:
                mineral_retreat += -mineral_distance_delta
                frames_without_mineral_progress += 1
            else:
                frames_without_mineral_progress += 1

            if (
                frames_without_mineral_progress > 30
                and target_distance_after / max_distance > FAR_FROM_MINERAL_DISTANCE
            ):
                indecision += 0.25

            mineral_heading_alignment += math.cos(relative_angle_to(ship, closest_mineral))
            best_distance = mineral_best_distances[closest_mineral]
            if target_distance_after < best_distance:
                mineral_progress += best_distance - target_distance_after
                mineral_best_distances[closest_mineral] = target_distance_after
        if output[2] > 0.5:  # Mine
            ship.mine(minerals)
            if len(minerals) < 3:  # Replenish minerals
                minerals.extend(Mineral() for _ in range(2))
            mineral_best_distances = {
                mineral: distance
                for mineral, distance in mineral_best_distances.items()
                if mineral in minerals
            }

        # Asteroids are part of the simulation, not just the visualization.
        # Move them during both headless training and rendered playback so
        # genomes are evaluated against the same task they are displayed in.
        for asteroid in asteroids:
            asteroid.move()
        
        # Termination conditions
        asteroid_collision = any(
            distance_between(ship, asteroid) < ship.radius + asteroid.radius
            for asteroid in asteroids
        )
        out_of_fuel = ship.fuel <= 0
        no_minerals_left = not minerals and ship.minerals == 0

        genome.fitness, fitness_components = calculate_fitness(
            ship,
            alive_time,
            mineral_progress,
            mineral_approach,
            mineral_heading_alignment,
            idle_time,
            asteroid_collision,
            cumulative_steering,
            asteroid_proximity,
            fitness_weights,
            mineral_retreat,
            indecision,
            return_components=True
        )
        
        # Visualization
        if visualizer:
            display_closest_mineral = min(
                (m for m in minerals),
                key=lambda m: distance_between(ship, m),
                default=None
            )
            display_closest_asteroid = min(
                (a for a in asteroids),
                key=lambda a: distance_between(ship, a),
                default=None
            )
            overlay_metrics = {
                "alive_time": alive_time,
                "nearest_mineral_distance": (
                    distance_between(ship, display_closest_mineral)
                    if display_closest_mineral else 0
                ),
                "nearest_asteroid_distance": (
                    distance_between(ship, display_closest_asteroid)
                    if display_closest_asteroid else 0
                ),
                "mineral_progress": mineral_progress,
                "mineral_approach": mineral_approach,
                "mineral_retreat": mineral_retreat,
                "idle_time": idle_time,
                "indecision": indecision,
                "cumulative_steering": cumulative_steering,
                "asteroid_proximity": asteroid_proximity,
                "asteroid_collision": asteroid_collision,
                "fuel_used": ship.fuel_used,
                "fitness_components": fitness_components,
            }
            screen.fill(BLACK)
            for mineral in minerals:
                mineral.draw()
            for asteroid in asteroids:
                asteroid.draw()
            ship.draw()
            visualizer.draw_stats(
                screen,
                genome.fitness,
                ship.minerals,
                ship.fuel,
                overlay_metrics
            )
            pygame.display.flip()
            clock.tick(30)
        
        if asteroid_collision or out_of_fuel or no_minerals_left or alive_time >= 5000:
            break

class TrainingVisualizer:
    def __init__(self):
        self.best_fitness = -float('inf')
        self.generation = 0
        self.start_time = time.time()
        self.font = pygame.font.SysFont(None, 36)
        self.small_font = pygame.font.SysFont(None, 22)
        
    def update_generation(self, best_genome):
        self.generation += 1
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            print(f"🔥 New best fitness: {self.best_fitness:.1f}")
        print(f"Generation {self.generation} best: {best_genome.fitness:.1f}")

    def draw_stats(self, screen, fitness, minerals, fuel, metrics=None):
        stats = [
            f"Gen: {self.generation}",
            f"Fitness: {fitness:.1f}",
            f"Best: {self.best_fitness:.1f}",
            f"Minerals: {minerals}",
            f"Fuel: {fuel:.1f}"
        ]

        if metrics:
            components = metrics["fitness_components"]
            stats.extend([
                f"Alive: {metrics['alive_time']}",
                f"Near mineral: {metrics['nearest_mineral_distance']:.1f}",
                f"Near asteroid: {metrics['nearest_asteroid_distance']:.1f}",
                f"Progress+: {metrics['mineral_progress']:.1f}",
                f"Approach+: {metrics['mineral_approach']:.1f}",
                f"Retreat-: {metrics['mineral_retreat']:.1f}",
                f"Idle: {metrics['idle_time']}",
                f"Indecision-: {metrics['indecision']:.1f}",
                f"Steering: {metrics['cumulative_steering']:.1f}",
                f"Ast danger-: {metrics['asteroid_proximity']:.1f}",
                f"Fuel used: {metrics['fuel_used']:.1f}",
                f"Hit asteroid: {metrics['asteroid_collision']}",
                f"Fit minerals: {components['minerals']:+.1f}",
                f"Fit approach: {components['approach']:+.1f}",
                f"Fit retreat: {components['retreat']:+.1f}",
                f"Fit idle: {components['idle']:+.1f}",
                f"Fit indecision: {components['indecision']:+.1f}",
                f"Fit asteroid: {components['asteroid_proximity'] + components['collision']:+.1f}",
            ])

        overlay_height = 18 + 28 * 5 + max(0, len(stats) - 5) * 20
        overlay = pygame.Surface((315, overlay_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (6, 6))

        for i, stat in enumerate(stats):
            font = self.font if i < 5 else self.small_font
            line_height = 28 if i < 5 else 20
            y_offset = 10 + i * 28 if i < 5 else 10 + 5 * 28 + (i - 5) * line_height
            text = font.render(stat, True, WHITE)
            screen.blit(text, (10, y_offset))

def eval_genomes(genomes, config):
    visualizer = config.visualizer
    
    # First evaluate all genomes to find the best
    best_in_generation = None
    best_fitness = -float('inf')
    
    for genome_id, genome in genomes:
        run_simulation(genome, config, visualizer=None)  # No visualization during evaluation
        print(genome_id,genome.fitness)
        # Track the best in this generation
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_in_generation = genome
    
    # Update visualizer with this generation's results
    visualizer.update_generation(best_in_generation)
    
    # Visualize the best genome from this generation
    if best_in_generation:
        print(f"Displaying generation {visualizer.generation} best (Fitness: {best_fitness:.1f})")
        run_simulation(best_in_generation, config, visualizer=visualizer)  # With visualization


def run_neat(config_file):
    # Initialize pygame
    pygame.init()
    global screen, clock, WIDTH, HEIGHT
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("NEAT - Space Miner Training")
    clock = pygame.time.Clock()
    
    # Create and store visualizer in config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)
    config.fitness_weights = load_fitness_weights(config_file)
    config.visualizer = TrainingVisualizer()
    
    # Create population
    population = neat.Population(config)
    
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Run NEAT
    try:
        winner = population.run(eval_genomes, GENERATIONS)
        print("\nTraining complete! Final best genome:")
        print(f"Fitness: {winner.fitness:.1f}")
        print(f"Nodes: {len(winner.nodes)}")
        print(f"Connections: {len(winner.connections)}")
        
        return winner
    finally:
        pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    winner = run_neat(config_file)
    # save the best genome
    with open("winner.pkl", "wb") as f:
        pickle.dump(winner, f)
