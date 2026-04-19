import numpy as np


class ArtificialBeeColony:
    def __init__(
        self,
        objective_function,
        num_food_sources=20,
        num_dimensions=2,
        max_iterations=100,
        limit=10,
        bounds=(-5, 5)
    ):
        """
        Initialize the Artificial Bee Colony (ABC) algorithm.

        Parameters:
            objective_function (callable): The objective function to minimize.
            num_food_sources (int): Number of food sources (candidate solutions).
            num_dimensions (int): Number of dimensions in the search space.
            max_iterations (int): Maximum number of iterations.
            limit (int): Abandonment limit for scout bees.
            bounds:
                Either:
                    - a tuple of scalars: (low, high)
                    - a list/tuple of per-dimension bounds:
                      [(low1, high1), (low2, high2), ...]
        """
        self.objective_function = objective_function
        self.num_food_sources = num_food_sources
        self.num_dimensions = num_dimensions
        self.max_iterations = max_iterations
        self.limit = limit

        # Parse bounds into per-dimension arrays
        self.lower_bounds, self.upper_bounds = self._parse_bounds(bounds, num_dimensions)

        # Initialize food sources
        self.food_sources = np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.num_food_sources, self.num_dimensions)
        )

        self.fitness_values = np.zeros(self.num_food_sources, dtype=float)
        self.trials = np.zeros(self.num_food_sources, dtype=int)
        self.best_solution = None
        self.best_fitness = float("inf")
        self.history = []

    def _parse_bounds(self, bounds, num_dimensions):
        """
        Convert bounds into two arrays:
            lower_bounds.shape == (num_dimensions,)
            upper_bounds.shape == (num_dimensions,)
        """
        # Case 1: scalar bounds like (-5, 5)
        if (
            isinstance(bounds, (tuple, list))
            and len(bounds) == 2
            and np.isscalar(bounds[0])
            and np.isscalar(bounds[1])
        ):
            low, high = bounds
            lower_bounds = np.full(num_dimensions, low, dtype=float)
            upper_bounds = np.full(num_dimensions, high, dtype=float)

        # Case 2: per-dimension bounds like [(-5, 5), (-5, 5), (-5, 5)]
        elif isinstance(bounds, (tuple, list)) and len(bounds) == num_dimensions:
            lower_bounds = np.array([b[0] for b in bounds], dtype=float)
            upper_bounds = np.array([b[1] for b in bounds], dtype=float)

        else:
            raise ValueError(
                f"Invalid bounds format for num_dimensions={num_dimensions}. "
                f"Use either (low, high) or a list of {num_dimensions} (low, high) pairs."
            )

        if np.any(lower_bounds >= upper_bounds):
            raise ValueError("Each lower bound must be smaller than its upper bound.")

        return lower_bounds, upper_bounds

    def fitness(self, f):
        """
        Convert objective value into fitness value.
        Higher fitness is better.
        """
        return 1 / (1 + f) if f >= 0 else 1 + abs(f)

    def evaluate_fitness(self):
        """
        Evaluate all food sources and update global best.
        """
        for i in range(self.num_food_sources):
            f = self.objective_function(self.food_sources[i])
            self.fitness_values[i] = self.fitness(f)

            if f < self.best_fitness:
                self.best_fitness = f
                self.best_solution = self.food_sources[i].copy()

    def employed_bee_phase(self):
        """
        Employed bee phase: Explore new solutions around existing food sources.
        """
        for i in range(self.num_food_sources):
            k = np.random.choice([x for x in range(self.num_food_sources) if x != i])
            j = np.random.randint(0, self.num_dimensions)

            phi = np.random.uniform(-1, 1)

            new_solution = self.food_sources[i].copy()
            new_solution[j] = (
                self.food_sources[i][j]
                + phi * (self.food_sources[i][j] - self.food_sources[k][j])
            )

            # Clip using per-dimension bounds
            new_solution = np.clip(new_solution, self.lower_bounds, self.upper_bounds)

            f_new = self.objective_function(new_solution)
            fitness_new = self.fitness(f_new)

            # Greedy selection
            if fitness_new > self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = fitness_new
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def onlooker_bee_phase(self):
        """
        Onlooker bee phase: Select food sources based on probability
        and further exploit promising areas.
        """
        total_fitness = np.sum(self.fitness_values)

        # Avoid division by zero
        if total_fitness == 0:
            probabilities = np.full(self.num_food_sources, 1 / self.num_food_sources)
        else:
            probabilities = self.fitness_values / total_fitness

        for _ in range(self.num_food_sources):
            i = np.random.choice(range(self.num_food_sources), p=probabilities)

            k = np.random.choice([x for x in range(self.num_food_sources) if x != i])
            j = np.random.randint(0, self.num_dimensions)

            phi = np.random.uniform(-1, 1)

            new_solution = self.food_sources[i].copy()
            new_solution[j] = (
                self.food_sources[i][j]
                + phi * (self.food_sources[i][j] - self.food_sources[k][j])
            )

            # Clip using per-dimension bounds
            new_solution = np.clip(new_solution, self.lower_bounds, self.upper_bounds)

            f_new = self.objective_function(new_solution)
            fitness_new = self.fitness(f_new)

            if fitness_new > self.fitness_values[i]:
                self.food_sources[i] = new_solution
                self.fitness_values[i] = fitness_new
                self.trials[i] = 0
            else:
                self.trials[i] += 1

    def scout_bee_phase(self):
        """
        Scout bee phase: Replace abandoned food sources with random solutions.
        """
        for i in range(self.num_food_sources):
            if self.trials[i] > self.limit:
                self.food_sources[i] = np.random.uniform(
                    low=self.lower_bounds,
                    high=self.upper_bounds,
                    size=self.num_dimensions
                )

                f_new = self.objective_function(self.food_sources[i])
                self.fitness_values[i] = self.fitness(f_new)
                self.trials[i] = 0

                if f_new < self.best_fitness:
                    self.best_fitness = f_new
                    self.best_solution = self.food_sources[i].copy()

    def optimize(self):
        """
        Run the ABC optimization algorithm.

        Returns:
            tuple:
                best_solution (np.ndarray)
                best_fitness (float)
                history (list)
        """
        self.evaluate_fitness()
        self.history = []

        for _ in range(self.max_iterations):
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            self.evaluate_fitness()

            self.history.append(self.best_fitness)

        return self.best_solution, self.best_fitness, self.history

    def run(self, max_iterations=None):
        """
        Compatibility wrapper for code that calls optimizer.run(...)

        Returns:
            tuple: (best_solution, best_fitness)
        """
        if max_iterations is not None:
            self.max_iterations = max_iterations

        best_solution, best_fitness, history = self.optimize()
        self.history = history
        return best_solution, best_fitness