import numpy as np


class Particle:
    def __init__(self, dim, bounds, func_name="rastrigin", max_velocity=1.0):
        self.func_name = func_name
        self.dim = dim

        # Parse bounds into per-dimension arrays
        self.lower_bounds, self.upper_bounds = self._parse_bounds(bounds, dim)

        # Velocity can be scalar or per-dimension array
        if np.isscalar(max_velocity):
            self.max_velocity = np.full(dim, float(max_velocity), dtype=float)
        else:
            max_velocity = np.asarray(max_velocity, dtype=float)
            if max_velocity.shape != (dim,):
                raise ValueError("max_velocity must be a scalar or have shape (dim,)")
            self.max_velocity = max_velocity

        # Current random position
        self.position = np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=self.dim
        )

        # Initial random velocity
        self.velocity = np.random.uniform(
            low=-self.max_velocity,
            high=self.max_velocity,
            size=self.dim
        )

        # Personal best
        self.p_best = self.position.copy()
        self.p_best_fitness = self.fitness(self.position)

    def _parse_bounds(self, bounds, dim):
        """
        Convert bounds into:
            lower_bounds.shape == (dim,)
            upper_bounds.shape == (dim,)
        """
        # Case 1: scalar bounds like (-5, 5)
        if (
            isinstance(bounds, (tuple, list))
            and len(bounds) == 2
            and np.isscalar(bounds[0])
            and np.isscalar(bounds[1])
        ):
            low, high = bounds
            lower_bounds = np.full(dim, low, dtype=float)
            upper_bounds = np.full(dim, high, dtype=float)

        # Case 2: per-dimension bounds like [(-5, 5), (-5, 5), (-5, 5)]
        elif isinstance(bounds, (tuple, list)) and len(bounds) == dim:
            lower_bounds = np.array([b[0] for b in bounds], dtype=float)
            upper_bounds = np.array([b[1] for b in bounds], dtype=float)

        else:
            raise ValueError(
                f"Invalid bounds format for dim={dim}. "
                f"Use either (low, high) or a list of {dim} (low, high) pairs."
            )

        if np.any(lower_bounds >= upper_bounds):
            raise ValueError("Each lower bound must be smaller than its upper bound.")

        return lower_bounds, upper_bounds

    def fitness(self, pos: np.ndarray) -> float:
        pos = np.asarray(pos, dtype=float)

        if self.func_name == "rastrigin":
            n = pos.size
            return float(10.0 * n + np.sum(pos**2 - 10.0 * np.cos(2.0 * np.pi * pos)))

        elif self.func_name == "ackley":
            n = pos.size
            sum_sq = np.sum(pos**2)
            sum_cos = np.sum(np.cos(2.0 * np.pi * pos))
            term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
            term2 = -np.exp(sum_cos / n)
            return float(term1 + term2 + 20.0 + np.e)

        elif self.func_name == "schwefel":
            n = pos.size
            return float(418.9829 * n - np.sum(pos * np.sin(np.sqrt(np.abs(pos)))))

        elif self.func_name == "rosenbrock":
            return float(np.sum(100.0 * (pos[1:] - pos[:-1] ** 2) ** 2 + (1.0 - pos[:-1]) ** 2))

        else:
            raise ValueError(f"Unsupported function name: {self.func_name}")

    def update(self, g_best, c1=1.0, c2=1.0, w=0.7):
        r1 = np.random.random(self.position.size)
        r2 = np.random.random(self.position.size)

        # Vectorized velocity update
        self.velocity = (
            w * self.velocity
            + c1 * r1 * (self.p_best - self.position)
            + c2 * r2 * (g_best - self.position)
        )

        # Clip velocity per dimension
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)

        # Update and clip position per dimension
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, self.lower_bounds, self.upper_bounds)

        # Update personal best
        current_fitness = self.fitness(self.position)
        if current_fitness < self.p_best_fitness:
            self.p_best = self.position.copy()
            self.p_best_fitness = current_fitness


class Swarm:
    def __init__(self, num_particles, dim, bounds, func_name="rastrigin"):
        self.dim = dim
        self.bounds = bounds
        self.func_name = func_name
        self.particles = []

        # Parse bounds once for swarm-wide use
        self.lower_bounds, self.upper_bounds = self._parse_bounds(bounds, dim)

        # Dynamic max velocity = 20% of search range per dimension
        search_width = self.upper_bounds - self.lower_bounds
        max_velocity = 0.2 * search_width

        for _ in range(num_particles):
            self.particles.append(
                Particle(
                    dim=dim,
                    bounds=bounds,
                    func_name=func_name,
                    max_velocity=max_velocity
                )
            )

        self.g_best = None
        self.g_best_fitness = float("inf")
        self.history = []

        self.find_global_best()

    def _parse_bounds(self, bounds, dim):
        """
        Same bounds parser as in Particle, kept here for swarm-level calculations.
        """
        if (
            isinstance(bounds, (tuple, list))
            and len(bounds) == 2
            and np.isscalar(bounds[0])
            and np.isscalar(bounds[1])
        ):
            low, high = bounds
            lower_bounds = np.full(dim, low, dtype=float)
            upper_bounds = np.full(dim, high, dtype=float)

        elif isinstance(bounds, (tuple, list)) and len(bounds) == dim:
            lower_bounds = np.array([b[0] for b in bounds], dtype=float)
            upper_bounds = np.array([b[1] for b in bounds], dtype=float)

        else:
            raise ValueError(
                f"Invalid bounds format for dim={dim}. "
                f"Use either (low, high) or a list of {dim} (low, high) pairs."
            )

        if np.any(lower_bounds >= upper_bounds):
            raise ValueError("Each lower bound must be smaller than its upper bound.")

        return lower_bounds, upper_bounds

    def find_global_best(self):
        for p in self.particles:
            if p.p_best_fitness < self.g_best_fitness:
                self.g_best_fitness = p.p_best_fitness
                self.g_best = p.p_best.copy()
        return self.g_best

    def update(self, c1=1.0, c2=1.0, w=0.7):
        # Get current global best before updating particles
        g_best = self.find_global_best()

        for p in self.particles:
            p.update(g_best, c1=c1, c2=c2, w=w)

        # Recompute global best after all particles move
        self.g_best_fitness = float("inf")
        self.g_best = None
        self.find_global_best()

    def simulate(self, steps, c1=1.0, c2=1.0, w=0.7):
        """
        Generator function to yield the swarm state at each step.
        """
        self.history = []

        for i in range(steps):
            self.update(c1=c1, c2=c2, w=w)
            self.history.append(self.g_best_fitness)

            yield {
                "iteration": i,
                "g_best": self.g_best.copy(),
                "g_best_fitness": self.g_best_fitness,
                "positions": [p.position.copy() for p in self.particles]
            }

    def run(self, steps, c1=1.0, c2=1.0, w=0.7):
        """
        Convenience method like the ABC version.

        Returns:
            tuple: (best_position, best_fitness)
        """
        for _ in self.simulate(steps, c1=c1, c2=c2, w=w):
            pass
        return self.g_best.copy(), self.g_best_fitness