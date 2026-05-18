"""
Genetic Algorithm for A/C Scheduling Optimization
Solves the mixed integer-continuous optimization problem for daily A/C operation scheduling
with Taipei's tiered electricity pricing.
"""

import numpy as np
from typing import Tuple, Dict, List, Callable
import warnings
from dataclasses import dataclass
from energy_consumption_utils import EnergyConsumptionCalculator
from energy_cost_utils import TaipeiTieredPricing


@dataclass
class GAConfig:
    """Configuration parameters for the genetic algorithm."""
    
    population_size: int = 100
    generations: int = 200
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 5
    
    # Objective function weights
    weight_cost: float = 1.0          # α: weight for electricity cost
    weight_consumption: float = 0.5   # β: weight for energy consumption
    weight_discomfort: float = 1.0    # δ: weight for discomfort penalty
    
    # Constraint parameters
    min_setpoint: float = 22.0
    max_setpoint: float = 27.0
    min_comfortable_temp: float = 18.0
    max_comfortable_temp: float = 28.0
    ac_power: float = 2.9


class GeneticAlgorithmAC:
    """
    Genetic Algorithm optimizer for A/C scheduling.
    
    Optimizes a 48-dimensional problem:
    - 24 binary variables (A/C on/off for each hour)
    - 24 continuous variables (temperature setpoints for each hour)
    
    Parameters:
    -----------
    outdoor_temps : np.ndarray
        Array of outdoor temperatures for each hour (shape: 24)
    config : GAConfig
        Configuration object with GA parameters
    """
    
    def __init__(self, outdoor_temps: np.ndarray, config: GAConfig = None):
        """
        Initialize the genetic algorithm.
        
        Parameters:
        -----------
        outdoor_temps : np.ndarray
            Array of outdoor temperatures for 24 hours
        config : GAConfig
            Configuration parameters (uses defaults if None)
        """
        if len(outdoor_temps) != 24:
            raise ValueError("outdoor_temps must have exactly 24 elements (one per hour)")
        
        self.outdoor_temps = np.array(outdoor_temps, dtype=float)
        self.config = config if config is not None else GAConfig()
        
        # Initialize utility calculators
        self.energy_calc = EnergyConsumptionCalculator(AC_power=self.config.ac_power)
        self.pricing = TaipeiTieredPricing(season='summer')
        
        # History tracking
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.best_individual_history = []
        
    def _create_individual(self) -> np.ndarray:
        """
        Create a random individual for the initial population.
        
        Returns:
        --------
        np.ndarray
            Individual with shape (48,):
            - Elements [0, 2, 4, ..., 46]: Binary A/C modes (0 or 1)
            - Elements [1, 3, 5, ..., 47]: Continuous setpoints
        """
        individual = np.zeros(48)
        
        # Binary modes: randomly 0 or 1
        individual[::2] = np.random.randint(0, 2, size=24)
        
        # Continuous setpoints: uniform in [min_setpoint, max_setpoint]
        individual[1::2] = np.random.uniform(
            self.config.min_setpoint,
            self.config.max_setpoint,
            size=24
        )
        
        return individual
    
    def _decode_individual(self, individual: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode an individual into modes and setpoints.
        
        Parameters:
        -----------
        individual : np.ndarray
            Individual array of shape (48,)
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            - modes: Binary array of shape (24,)
            - setpoints: Continuous array of shape (24,)
        """
        modes = individual[::2].astype(int)
        setpoints = individual[1::2]
        
        # Clip setpoints to valid range
        setpoints = np.clip(
            setpoints,
            self.config.min_setpoint,
            self.config.max_setpoint
        )
        
        return modes, setpoints
    
    def _evaluate_fitness(self, individual: np.ndarray) -> float:
        """
        Evaluate the fitness (objective function) of an individual.
        
        Lower fitness is better (minimization problem).
        
        Parameters:
        -----------
        individual : np.ndarray
            Individual array of shape (48,)
        
        Returns:
        --------
        float
            Fitness value (negative = better performance)
        """
        modes, setpoints = self._decode_individual(individual)
        
        # Calculate energy metrics
        indoor_temps, total_consumption, total_discomfort = \
            self.energy_calc.calculate_daily_metrics(modes, setpoints, self.outdoor_temps)
        
        # Calculate cost
        total_cost = self.pricing.calculate_cost(total_consumption)
        
        # Objective function: weighted sum of three objectives
        # F(x) = α * C_total + β * E_total + δ * D_total
        fitness = (
            self.config.weight_cost * total_cost +
            self.config.weight_consumption * total_consumption +
            self.config.weight_discomfort * total_discomfort
        )
        
        return fitness
    
    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Select an individual using tournament selection.
        
        Parameters:
        -----------
        population : np.ndarray
            Population array of shape (pop_size, 48)
        fitness : np.ndarray
            Fitness values for all individuals
        
        Returns:
        --------
        np.ndarray
            Selected individual
        """
        tournament_indices = np.random.choice(
            len(population),
            size=self.config.tournament_size,
            replace=False
        )
        best_tournament_idx = tournament_indices[
            np.argmin(fitness[tournament_indices])
        ]
        
        return population[best_tournament_idx].copy()
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform uniform crossover between two parents.
        
        Parameters:
        -----------
        parent1 : np.ndarray
            First parent individual
        parent2 : np.ndarray
            Second parent individual
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Two offspring individuals
        """
        if np.random.rand() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover: each gene randomly chosen from either parent
        mask = np.random.randint(0, 2, size=len(parent1))
        
        offspring1 = np.where(mask, parent1, parent2)
        offspring2 = np.where(mask, parent2, parent1)
        
        return offspring1, offspring2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutate an individual.
        
        Binary genes (modes) are flipped with probability mutation_rate.
        Continuous genes (setpoints) are perturbed with Gaussian noise.
        
        Parameters:
        -----------
        individual : np.ndarray
            Individual to mutate
        
        Returns:
        --------
        np.ndarray
            Mutated individual
        """
        mutant = individual.copy()
        
        # Mutate binary genes (A/C modes) - flip with probability
        for i in range(0, 48, 2):  # Even indices: binary modes
            if np.random.rand() < self.config.mutation_rate:
                mutant[i] = 1 - mutant[i]
        
        # Mutate continuous genes (setpoints) - Gaussian perturbation
        for i in range(1, 48, 2):  # Odd indices: continuous setpoints
            if np.random.rand() < self.config.mutation_rate:
                # Gaussian mutation with std = 0.5°C
                mutant[i] += np.random.normal(0, 0.5)
                # Clip to valid range
                mutant[i] = np.clip(
                    mutant[i],
                    self.config.min_setpoint,
                    self.config.max_setpoint
                )
        
        return mutant
    
    def optimize(self, random_seed: int = None) -> Dict:
        """
        Run the genetic algorithm optimization.
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducibility (optional)
        
        Returns:
        --------
        dict
            Results dictionary with keys:
            - 'best_individual': Best solution found
            - 'best_fitness': Fitness of best solution
            - 'best_modes': A/C modes for best solution
            - 'best_setpoints': Temperature setpoints for best solution
            - 'best_metrics': Detailed metrics for best solution
            - 'fitness_history': Best and mean fitness per generation
            - 'population': Final population
            - 'fitness': Final fitness values
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize population
        population = np.array([self._create_individual() for _ in range(self.config.population_size)])
        
        # Evaluation loop
        for generation in range(self.config.generations):
            # Evaluate fitness
            fitness = np.array([self._evaluate_fitness(ind) for ind in population])
            
            # Track history
            best_idx = np.argmin(fitness)
            best_fitness = fitness[best_idx]
            mean_fitness = np.mean(fitness)
            
            self.best_fitness_history.append(best_fitness)
            self.mean_fitness_history.append(mean_fitness)
            self.best_individual_history.append(population[best_idx].copy())
            
            # Progress reporting
            if (generation + 1) % 50 == 0:
                print(f"Generation {generation + 1}/{self.config.generations} "
                      f"| Best fitness: {best_fitness:.2f} "
                      f"| Mean fitness: {mean_fitness:.2f}")
            
            # Selection: keep elite individuals
            elite_indices = np.argsort(fitness)[:self.config.elite_size]
            elite_population = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()
            
            # Create new population
            new_population = [elite_population[i].copy() for i in range(self.config.elite_size)]
            
            # Generate offspring through crossover and mutation
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                
                new_population.append(offspring1)
                if len(new_population) < self.config.population_size:
                    new_population.append(offspring2)
            
            population = np.array(new_population[:self.config.population_size])
        
        # Final evaluation
        final_fitness = np.array([self._evaluate_fitness(ind) for ind in population])
        best_idx = np.argmin(final_fitness)
        best_individual = population[best_idx]
        best_fitness = final_fitness[best_idx]
        
        # Decode best solution
        best_modes, best_setpoints = self._decode_individual(best_individual)
        best_metrics = self._get_solution_metrics(best_modes, best_setpoints)
        
        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'best_modes': best_modes,
            'best_setpoints': best_setpoints,
            'best_metrics': best_metrics,
            'fitness_history': {
                'best': self.best_fitness_history,
                'mean': self.mean_fitness_history,
            },
            'population': population,
            'fitness': final_fitness,
        }
    
    def _get_solution_metrics(self, modes: np.ndarray, setpoints: np.ndarray) -> Dict:
        """
        Get detailed metrics for a solution.
        
        Parameters:
        -----------
        modes : np.ndarray
            A/C modes for each hour
        setpoints : np.ndarray
            Temperature setpoints for each hour
        
        Returns:
        --------
        dict
            Detailed metrics including energy, cost, comfort, and thermal data
        """
        indoor_temps, total_consumption, total_discomfort = \
            self.energy_calc.calculate_daily_metrics(modes, setpoints, self.outdoor_temps)
        
        total_cost = self.pricing.calculate_cost(total_consumption)
        cost_breakdown = self.pricing.get_cost_breakdown(total_consumption)
        
        return {
            'indoor_temperatures': indoor_temps,
            'total_consumption_kwh': total_consumption,
            'total_cost_twd': total_cost,
            'total_discomfort': total_discomfort,
            'ac_on_hours': int(np.sum(modes)),
            'mean_indoor_temp': np.mean(indoor_temps),
            'min_indoor_temp': np.min(indoor_temps),
            'max_indoor_temp': np.max(indoor_temps),
            'cost_breakdown': cost_breakdown,
        }


def run_optimization_scenario(
    outdoor_temps: np.ndarray,
    scenario_name: str,
    weight_cost: float = 1.0,
    weight_consumption: float = 0.5,
    weight_discomfort: float = 1.0,
    generations: int = 200,
    random_seed: int = 42
) -> Dict:
    """
    Run a complete optimization scenario with specified weights.
    
    Parameters:
    -----------
    outdoor_temps : np.ndarray
        Array of outdoor temperatures (24 hours)
    scenario_name : str
        Name of the scenario (for display)
    weight_cost : float
        Weight for electricity cost (α)
    weight_consumption : float
        Weight for energy consumption (β)
    weight_discomfort : float
        Weight for discomfort penalty (δ)
    generations : int
        Number of GA generations to run
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Results from the optimization
    """
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario_name}")
    print(f"Weights: Cost={weight_cost}, Consumption={weight_consumption}, Discomfort={weight_discomfort}")
    print(f"{'='*60}\n")
    
    config = GAConfig(
        weight_cost=weight_cost,
        weight_consumption=weight_consumption,
        weight_discomfort=weight_discomfort,
        generations=generations,
    )
    
    ga = GeneticAlgorithmAC(outdoor_temps, config)
    results = ga.optimize(random_seed=random_seed)
    
    print(f"\nOptimization complete for {scenario_name}")
    print(f"Best fitness: {results['best_fitness']:.2f}")
    print(f"Daily cost: {results['best_metrics']['total_cost_twd']:.2f} TWD")
    print(f"Daily consumption: {results['best_metrics']['total_consumption_kwh']:.2f} kWh")
    print(f"Daily discomfort: {results['best_metrics']['total_discomfort']:.2f}")
    print(f"A/C on hours: {results['best_metrics']['ac_on_hours']}/24")
    
    return results