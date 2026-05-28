import pygame
import random
import math
import os
import neat
import time
# POOR FITNESS FUNCTION - spin on own axis

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

# Game Classes (same as before)
class Spaceship:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.speed = 5
        self.angle = 0
        self.fuel = 100
        self.minerals = 0
        self.radius = 15

    def move(self, dx, dy):
        if self.fuel > 0:
            self.x = (self.x + dx) % WIDTH
            self.y = (self.y + dy) % HEIGHT
            self.fuel -= 0.1

    def mine(self, minerals):
        for mineral in minerals[:]:
            dist = math.hypot(self.x - mineral.x, self.y - mineral.y)
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
    def __init__(self):
        self.x = random.randint(0, WIDTH)
        self.y = random.randint(0, HEIGHT)
        self.radius = random.randint(15, 30)
        self.speed_x = random.uniform(-2, 2)
        self.speed_y = random.uniform(-2, 2)

    def move(self):
        self.x = (self.x + self.speed_x) % WIDTH
        self.y = (self.y + self.speed_y) % HEIGHT

    def draw(self):
        pygame.draw.circle(screen, RED, (int(self.x), int(self.y)), self.radius)

def run_simulation(genome, config, visualizer=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    ship = Spaceship()
    minerals = [Mineral() for _ in range(5)]
    asteroids = [Asteroid() for _ in range(8)]
    alive_time = 0
    
    while True:
        alive_time += 1
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Find closest objects
        closest_mineral = min((m for m in minerals), 
                            key=lambda m: math.hypot(ship.x-m.x, ship.y-m.y), 
                            default=None)
        closest_asteroid = min((a for a in asteroids), 
                              key=lambda a: math.hypot(ship.x-a.x, ship.y-a.y))
        
        # Get inputs (handle case where all minerals are collected)
        inputs = [
            math.hypot(ship.x - closest_mineral.x)/WIDTH if closest_mineral else 0,
            math.atan2(closest_mineral.y-ship.y, closest_mineral.x-ship.x)/math.pi if closest_mineral else 0,
            math.hypot(ship.x - closest_asteroid.x)/WIDTH,
            ship.fuel / 100.0
        ]
        
        # Get actions from network
        output = net.activate(inputs)
        
        # Execute actions
        ship.angle += (output[0] * 2 - 1) * 0.1  # Turn (-1 to 1)
        if output[1] > 0.5:  # Thrust
            dx = ship.speed * math.cos(ship.angle)
            dy = ship.speed * math.sin(ship.angle)
            ship.move(dx, dy)
        if output[2] > 0.5:  # Mine
            ship.mine(minerals)
            if len(minerals) < 3:  # Replenish minerals
                minerals.extend(Mineral() for _ in range(2))

        # Asteroids are part of the simulation, not just the visualization.
        # Move them during both headless training and rendered playback so
        # genomes are evaluated against the same task they are displayed in.
        for asteroid in asteroids:
            asteroid.move()
        
        # Calculate fitness - reward both survival and mining
        genome.fitness = ship.minerals * 10 + alive_time * 0.01  # Reduced time bonus
        
        # Visualization
        if visualizer:
            screen.fill(BLACK)
            for mineral in minerals:
                mineral.draw()
            for asteroid in asteroids:
                asteroid.draw()
            ship.draw()
            visualizer.draw_stats(screen, genome.fitness, ship.minerals, ship.fuel)
            pygame.display.flip()
            clock.tick(30)
        
        # Termination conditions
        asteroid_collision = any(
            math.hypot(ship.x - asteroid.x, ship.y - asteroid.y) < ship.radius + asteroid.radius
            for asteroid in asteroids
        )
        out_of_fuel = ship.fuel <= 0
        no_minerals_left = not minerals and ship.minerals == 0
        
        if asteroid_collision or out_of_fuel or no_minerals_left or alive_time >= 5000:
            break

class TrainingVisualizer:
    def __init__(self):
        self.best_fitness = -float('inf')
        self.generation = 0
        self.start_time = time.time()
        self.font = pygame.font.SysFont(None, 36)
        
    def update_generation(self, best_genome):
        self.generation += 1
        if best_genome.fitness > self.best_fitness:
            self.best_fitness = best_genome.fitness
            print(f"🔥 New best fitness: {self.best_fitness:.1f}")
        print(f"Generation {self.generation} best: {best_genome.fitness:.1f}")

    def draw_stats(self, screen, fitness, minerals, fuel):
        stats = [
            f"Gen: {self.generation}",
            f"Fitness: {fitness:.1f}",
            f"Best: {self.best_fitness:.1f}",
            f"Minerals: {minerals}",
            f"Fuel: {fuel:.1f}"
        ]
        
        for i, stat in enumerate(stats):
            text = self.font.render(stat, True, WHITE)
            screen.blit(text, (10, 10 + i * 40))

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
    config.visualizer = TrainingVisualizer()
    
    # Create population
    population = neat.Population(config)
    
    # Add reporters
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # Run NEAT
    try:
        winner = population.run(eval_genomes, 50)
        print("\nTraining complete! Final best genome:")
        print(f"Fitness: {winner.fitness:.1f}")
        print(f"Nodes: {len(winner.nodes)}")
        print(f"Connections: {len(winner.connections)}")
    finally:
        pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, "neat_config.txt")
    run_neat(config_file)
