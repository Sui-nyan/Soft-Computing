from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v, dtype=float)
    return v / norm


class RootStatus(Enum):
    ALIVE = auto()
    DEAD = auto()


@dataclass
class MoisturePatch:
    position: np.ndarray
    strength: float


class MoistureMap:
    """
    Dynamic moisture field.
    Positive patches attract roots.
    Negative patches represent local depletion from overuse.
    """

    def __init__(
        self,
        dimension: int,
        sigma: float = 0.75,
        base_moisture: float = 0.05,
        evaporation_rate: float = 0.01,
        max_patches: int = 500,
    ) -> None:
        self.dimension = int(dimension)
        self.sigma = float(sigma)
        self.base_moisture = float(base_moisture)
        self.evaporation_rate = float(evaporation_rate)
        self.max_patches = int(max_patches)
        self.patches: List[MoisturePatch] = []

    def moisture(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if not self.patches:
            return self.base_moisture

        positions = np.array([p.position for p in self.patches], dtype=float)
        strengths = np.array([p.strength for p in self.patches], dtype=float)

        diffs = x - positions
        sq_dists = np.sum(diffs * diffs, axis=1)
        kernels = np.exp(-sq_dists / (2.0 * self.sigma * self.sigma))

        value = self.base_moisture + float(np.sum(strengths * kernels))
        return max(value, 1e-8)

    def moisture_gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if not self.patches:
            return np.zeros(self.dimension, dtype=float)

        positions = np.array([p.position for p in self.patches], dtype=float)
        strengths = np.array([p.strength for p in self.patches], dtype=float)

        diffs = x - positions
        sq_dists = np.sum(diffs * diffs, axis=1)
        kernels = np.exp(-sq_dists / (2.0 * self.sigma * self.sigma))

        # Gradient of sum_i strength_i * exp(-||x-p_i||^2 / (2 sigma^2))
        grad = np.sum(
            ((-strengths * kernels)[:, None] * diffs) / (self.sigma * self.sigma),
            axis=0,
        )
        return grad

    def deposit(self, x: np.ndarray, amount: float) -> None:
        amount = float(amount)
        if amount <= 0.0:
            return
        self.patches.append(MoisturePatch(np.asarray(x, dtype=float).copy(), amount))
        self._trim_if_needed()

    def deplete(self, x: np.ndarray, amount: float) -> None:
        amount = float(amount)
        if amount <= 0.0:
            return
        self.patches.append(MoisturePatch(np.asarray(x, dtype=float).copy(), -amount))
        self._trim_if_needed()

    def evaporate(self) -> None:
        if not self.patches:
            return

        decay = 1.0 - self.evaporation_rate
        kept: List[MoisturePatch] = []

        for patch in self.patches:
            new_strength = patch.strength * decay
            if abs(new_strength) > 1e-6:
                kept.append(MoisturePatch(patch.position, new_strength))

        self.patches = kept

    def _trim_if_needed(self) -> None:
        if len(self.patches) <= self.max_patches:
            return
        self.patches.sort(key=lambda p: abs(p.strength), reverse=True)
        self.patches = self.patches[: self.max_patches]


@dataclass
class RootAgent:
    position: np.ndarray
    energy: float
    step_size: float
    step_cost: float
    max_children: int
    current_fitness: float

    personal_best_position: np.ndarray = field(init=False)
    personal_best_fitness: float = field(init=False)
    status: RootStatus = field(default=RootStatus.ALIVE, init=False)
    path: List[np.ndarray] = field(default_factory=list, init=False)
    last_direction: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).copy()
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float(self.current_fitness)
        self.path.append(self.position.copy())
        self.last_direction = np.zeros_like(self.position, dtype=float)

    def calculate_competition_repulsion(
        self,
        roots: Sequence["RootAgent"],
        repulsion_radius: float = 1.0,
        eps: float = 1e-12,
    ) -> np.ndarray:
        repulsion = np.zeros_like(self.position, dtype=float)

        for other in roots:
            if other is self or other.status is RootStatus.DEAD or other.energy <= 0.0:
                continue

            diff = self.position - other.position
            dist = np.linalg.norm(diff)

            if eps < dist < repulsion_radius:
                repulsion += (1.0 - dist / repulsion_radius) * (diff / dist)

        return repulsion

    def calculate_direction(
        self,
        roots: Sequence["RootAgent"],
        moisture_map: MoistureMap,
        gravity_vector: np.ndarray,
        weights: Tuple[float, float, float, float],
        repulsion_radius: float,
        rng: np.random.Generator,
        personal_best_weight: float,
    ) -> np.ndarray:
        weight_h, weight_g, weight_r, weight_c = weights

        h = normalize(moisture_map.moisture_gradient(self.position))
        g = normalize(gravity_vector)
        r = normalize(rng.normal(size=self.position.shape))
        c = normalize(
            self.calculate_competition_repulsion(
                roots=roots,
                repulsion_radius=repulsion_radius,
            )
        )
        b = normalize(self.personal_best_position - self.position)

        direction_vector = (
            weight_h * h
            + weight_g * g
            + weight_r * r
            + weight_c * c
            + personal_best_weight * b
        )

        if np.linalg.norm(direction_vector) < 1e-12:
            direction_vector = rng.normal(size=self.position.shape)

        direction = normalize(direction_vector)
        self.last_direction = direction
        return direction

    def step(
    self,
    roots: Sequence["RootAgent"],
    moisture_map: MoistureMap,
    objective_function: Callable[[np.ndarray], float],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    gravity_vector: np.ndarray,
    weights: Tuple[float, float, float, float],
    repulsion_radius: float,
    alpha: float,
    deposit_scale: float,
    depletion_scale: float,
    rng: np.random.Generator,
    personal_best_weight: float,
    ) -> None:
        if self.status is RootStatus.DEAD or self.energy <= 0.0:
            self.status = RootStatus.DEAD
            return

        prev_fitness = self.current_fitness

        direction = self.calculate_direction(
            roots=roots,
            moisture_map=moisture_map,
            gravity_vector=gravity_vector,
            weights=weights,
            repulsion_radius=repulsion_radius,
            rng=rng,
            personal_best_weight=personal_best_weight,
        )

        adaptive_step = self.step_size * max(0.25, min(2.0, self.energy / 5.0))

        proposed_position = self.position + direction * adaptive_step
        proposed_position = np.clip(proposed_position, lower_bounds, upper_bounds)

        new_fitness = float(objective_function(proposed_position))
        improvement = prev_fitness - new_fitness

        # Relative improvement instead of absolute improvement
        relative_improvement = improvement / (abs(prev_fitness) + 1e-12)
        relative_improvement = max(0.0, relative_improvement)
        relative_improvement = min(relative_improvement, 1.0)

        self.position = proposed_position
        self.current_fitness = new_fitness
        self.path.append(self.position.copy())

        # Energy update
        self.energy += alpha * relative_improvement - self.step_cost

        # Moisture deposit based on relative improvement
        if relative_improvement > 0.0:
            moisture_map.deposit(self.position, amount=deposit_scale * relative_improvement)

        # New personal best leaves a modest deposit
        if new_fitness < self.personal_best_fitness:
            self.personal_best_fitness = new_fitness
            self.personal_best_position = self.position.copy()
            moisture_map.deposit(self.position, amount=0.5 * deposit_scale)

        # Repeated visitation depletes the region
        moisture_map.deplete(self.position, amount=depletion_scale)

        if self.energy <= 0.0:
            self.status = RootStatus.DEAD

    def split(
        self,
        objective_function: Callable[[np.ndarray], float],
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        rng: np.random.Generator,
        split_ratio: float = 0.45,
        child_offset: float = 0.06,
    ) -> List["RootAgent"]:
        if self.status is RootStatus.DEAD or self.energy <= 0.0:
            return []

        n_children = max(1, int(self.max_children))

        total_child_energy = self.energy * split_ratio
        child_energy = total_child_energy / n_children
        self.energy *= (1.0 - split_ratio)

        base_dir = self.last_direction.copy()
        if np.linalg.norm(base_dir) < 1e-12:
            base_dir = normalize(rng.normal(size=self.position.shape))

        children: List[RootAgent] = []

        for _ in range(n_children):
            child_dir = normalize(base_dir + 0.30 * rng.normal(size=self.position.shape))
            child_pos = self.position + child_dir * child_offset
            child_pos = np.clip(child_pos, lower_bounds, upper_bounds)

            child_fit = float(objective_function(child_pos))
            child = RootAgent(
                position=child_pos,
                energy=child_energy,
                step_size=self.step_size,
                step_cost=self.step_cost,
                max_children=self.max_children,
                current_fitness=child_fit,
            )
            child.last_direction = child_dir
            children.append(child)

        if self.energy <= 0.0:
            self.status = RootStatus.DEAD

        return children


class PlantAlgorithm:
    """
    Plant-root-inspired optimizer for minimization.

    Main ideas:
    - roots move using hydrotropism, random exploration, competition repulsion,
      optional gravity, and attraction to their own personal best
    - roots gain energy when they improve the objective
    - roots lose energy through movement cost
    - high-energy roots can split into children
    - good regions receive moisture deposits
    - overused regions are depleted
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: Sequence[Tuple[float, float]],
        max_root_count: int = 60,
        initial_energy: float = 6.0,
        step_size: float = 0.12,
        step_cost: float = 0.03,
        splitting_threshold: float = 5.0,
        max_children: int = 2,
        alpha: float = 8.0,
        deposit_scale: float = 4.0,
        depletion_scale: float = 0.03,
        weights: Tuple[float, float, float, float] = (1.8, 0.0, 0.25, 0.45),
        repulsion_radius: float = 0.8,
        gravity_vector: Optional[np.ndarray] = None,
        personal_best_weight: float = 1.2,
        moisture_sigma: float = 0.9,
        moisture_base: float = 0.05,
        evaporation_rate: float = 0.01,
        split_ratio: float = 0.45,
        child_offset: float = 0.06,
        initial_root_count: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        self.objective_function = objective_function

        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be a sequence of (lower, upper) pairs")

        self.lower_bounds = self.bounds[:, 0]
        self.upper_bounds = self.bounds[:, 1]
        self.dimension = self.bounds.shape[0]
        self.history: List[float] = []

        if gravity_vector is None:
            gravity_vector = np.zeros(self.dimension, dtype=float)
        else:
            gravity_vector = np.asarray(gravity_vector, dtype=float)
            if gravity_vector.shape != (self.dimension,):
                raise ValueError("gravity_vector must have shape (dimension,)")

        self.max_root_count = int(max_root_count)
        self.initial_energy = float(initial_energy)
        self.step_size = float(step_size)
        self.step_cost = float(step_cost)
        self.splitting_threshold = float(splitting_threshold)
        self.max_children = int(max_children)
        self.alpha = float(alpha)
        self.deposit_scale = float(deposit_scale)
        self.depletion_scale = float(depletion_scale)
        self.weights = tuple(float(w) for w in weights)
        self.repulsion_radius = float(repulsion_radius)
        self.gravity_vector = gravity_vector
        self.personal_best_weight = float(personal_best_weight)
        self.split_ratio = float(split_ratio)
        self.child_offset = float(child_offset)
        self.initial_root_count = int(initial_root_count)

        self.rng = np.random.default_rng(seed)

        self.moisture_map = MoistureMap(
            dimension=self.dimension,
            sigma=moisture_sigma,
            base_moisture=moisture_base,
            evaporation_rate=evaporation_rate,
        )

        self.roots: List[RootAgent] = []
        self.best_visited_position: Optional[np.ndarray] = None
        self.best_visited_fitness: float = np.inf

        self._initialize_roots()

    def _sample_random_position(self) -> np.ndarray:
        return self.rng.uniform(self.lower_bounds, self.upper_bounds)

    def _initialize_roots(self) -> None:
        for _ in range(self.initial_root_count):
            position = self._sample_random_position()
            fitness = float(self.objective_function(position))

            root = RootAgent(
                position=position,
                energy=self.initial_energy,
                step_size=self.step_size,
                step_cost=self.step_cost,
                max_children=self.max_children,
                current_fitness=fitness,
            )
            self.roots.append(root)
            self._update_global_best(root.personal_best_position, root.personal_best_fitness)

    def _update_global_best(self, position: np.ndarray, fitness: float) -> None:
        if fitness < self.best_visited_fitness:
            self.best_visited_fitness = float(fitness)
            self.best_visited_position = np.asarray(position, dtype=float).copy()

    def optimize_step(self) -> None:
        self.moisture_map.evaporate()
        new_roots: List[RootAgent] = []

        alive_roots = [r for r in self.roots if r.status is RootStatus.ALIVE and r.energy > 0.0]
        alive_count = len(alive_roots)

        for root in list(self.roots):
            if root.status is RootStatus.DEAD or root.energy <= 0.0:
                root.status = RootStatus.DEAD
                continue

            current_population = alive_count + len(new_roots)

            if root.energy >= self.splitting_threshold and current_population < self.max_root_count:
                remaining_capacity = self.max_root_count - current_population
                child_count = min(root.max_children, remaining_capacity)

                if child_count > 0:
                    original_max_children = root.max_children
                    root.max_children = child_count

                    children = root.split(
                        objective_function=self.objective_function,
                        lower_bounds=self.lower_bounds,
                        upper_bounds=self.upper_bounds,
                        rng=self.rng,
                        split_ratio=self.split_ratio,
                        child_offset=self.child_offset,
                    )

                    root.max_children = original_max_children
                    new_roots.extend(children)

                    # Slight depletion at the split point to reduce immediate pile-up
                    self.moisture_map.deplete(root.position, amount=self.depletion_scale)
                else:
                    root.step(
                        roots=self.roots,
                        moisture_map=self.moisture_map,
                        objective_function=self.objective_function,
                        lower_bounds=self.lower_bounds,
                        upper_bounds=self.upper_bounds,
                        gravity_vector=self.gravity_vector,
                        weights=self.weights,
                        repulsion_radius=self.repulsion_radius,
                        alpha=self.alpha,
                        deposit_scale=self.deposit_scale,
                        depletion_scale=self.depletion_scale,
                        rng=self.rng,
                        personal_best_weight=self.personal_best_weight,
                    )
            else:
                root.step(
                    roots=self.roots,
                    moisture_map=self.moisture_map,
                    objective_function=self.objective_function,
                    lower_bounds=self.lower_bounds,
                    upper_bounds=self.upper_bounds,
                    gravity_vector=self.gravity_vector,
                    weights=self.weights,
                    repulsion_radius=self.repulsion_radius,
                    alpha=self.alpha,
                    deposit_scale=self.deposit_scale,
                    depletion_scale=self.depletion_scale,
                    rng=self.rng,
                    personal_best_weight=self.personal_best_weight,
                )

            self._update_global_best(root.personal_best_position, root.personal_best_fitness)

        self.roots.extend(new_roots)

        for child in new_roots:
            self._update_global_best(child.personal_best_position, child.personal_best_fitness)

    def run(self, max_iterations: int = 1200) -> Tuple[np.ndarray, float]:
        self.history = [self.best_visited_fitness]

        for _ in range(max_iterations):
            alive = any(r.status is RootStatus.ALIVE and r.energy > 0.0 for r in self.roots)
            if not alive:
                break
            self.optimize_step()
            self.history.append(self.best_visited_fitness)

        if self.best_visited_position is None:
            raise RuntimeError("No root was initialized.")

        return self.best_visited_position.copy(), float(self.best_visited_fitness)

    def get_best_visited_position(self) -> np.ndarray:
        if self.best_visited_position is None:
            raise RuntimeError("No best position available.")
        return self.best_visited_position.copy()

    def get_best_visited_fitness(self) -> float:
        return float(self.best_visited_fitness)

