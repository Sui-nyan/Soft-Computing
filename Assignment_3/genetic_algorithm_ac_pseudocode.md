# Pseudocode for `GeneticAlgorithmAC`

## Overview

The `GeneticAlgorithmAC` class optimizes an air-conditioning schedule using a genetic algorithm.  
Each solution, or chromosome, alternates between:

- A binary A/C mode gene: `0 = OFF`, `1 = ON`
- A continuous temperature setpoint gene

The fitness function minimizes a weighted combination of:

- Electricity cost
- Energy consumption
- Thermal discomfort

Lower fitness values represent better solutions.

---

## Class: `GeneticAlgorithmAC`

```text
CLASS GeneticAlgorithmAC
```

---

## Method: `__init__`

```text
METHOD __init__(outdoor_temps, config, thermal_props)

    Store outdoor temperatures as numeric array

    Set num_hours = number of temperature values
    Set chromosome_length = 2 * num_hours

    IF chromosome_length is not 2 * num_hours
        Raise assertion error

    IF outdoor_temps does not contain 672 values
        Raise error

    IF outdoor_temps is empty
        Raise error

    IF config is provided
        Use given config
    ELSE
        Use default GAConfig

    IF thermal_props is provided
        Use given thermal properties
    ELSE
        Use default ThermalProperties

    Create thermal simulator
    Create Taipei electricity pricing model

    Initialize:
        best_fitness_history as empty list
        mean_fitness_history as empty list
        best_individual_history as empty list
        evaluations = 0
```

---

## Method: `_create_individual`

```text
METHOD _create_individual()

    Create an empty chromosome of length chromosome_length

    FOR every even index in chromosome
        Randomly assign 0 or 1
        // Represents A/C OFF or ON

    FOR every odd index in chromosome
        Randomly assign a setpoint temperature
        between min_setpoint and max_setpoint

    RETURN chromosome
```

---

## Method: `_decode_individual`

```text
METHOD _decode_individual(individual)

    Extract A/C modes from even indices
    Convert modes to integers

    Extract temperature setpoints from odd indices

    Clip setpoints so they remain within:
        min_setpoint <= setpoint <= max_setpoint

    RETURN modes, setpoints
```

---

## Method: `_evaluate_fitness`

```text
METHOD _evaluate_fitness(individual)

    Decode individual into:
        modes
        setpoints

    Run thermal simulation using:
        modes
        setpoints
        outdoor temperatures
        initial indoor temperature

    Extract from simulation:
        total energy consumption
        total discomfort

    Calculate electricity cost from total energy

    Compute fitness as weighted sum:

        fitness =
            weight_cost * total_cost
            + weight_consumption * total_energy
            + weight_discomfort * total_discomfort

    Increment number of evaluations

    RETURN fitness
```

---

## Method: `_tournament_selection`

```text
METHOD _tournament_selection(population, fitness)

    Randomly choose tournament_size individuals from population

    Find the individual in the tournament with the lowest fitness

    RETURN copy of that individual
```

---

## Method: `_crossover`

```text
METHOD _crossover(parent1, parent2)

    Generate random number

    IF random number is greater than crossover_rate
        RETURN copies of parent1 and parent2

    Create random binary mask with same length as parents

    offspring1 gets each gene from:
        parent1 where mask is true
        parent2 where mask is false

    offspring2 gets each gene from:
        parent2 where mask is true
        parent1 where mask is false

    RETURN offspring1, offspring2
```

---

## Method: `_mutate`

```text
METHOD _mutate(individual)

    Copy individual into mutant

    FOR each even-indexed gene
        // A/C mode gene

        IF random number < mutation_rate
            Flip the binary value:
                0 becomes 1
                1 becomes 0

    FOR each odd-indexed gene
        // Temperature setpoint gene

        IF random number < mutation_rate
            Add Gaussian noise to the setpoint

            Clip the result so it remains within:
                min_setpoint and max_setpoint

    RETURN mutant
```

---

## Method: `optimize`

```text
METHOD optimize(random_seed, verbose)

    IF random_seed is provided
        Set random seed

    Reset evaluations to 0

    Create initial population:
        Repeat population_size times:
            create a random individual

    IF verbose
        Print optimization settings

    FOR each generation

        Evaluate fitness for every individual in population

        Find best individual in current generation
        Find best fitness
        Find mean fitness

        Save:
            best fitness
            mean fitness
            copy of best individual

        IF verbose and generation is a reporting generation
            Print progress

        Select elite individuals with lowest fitness

        Start new population with elite individuals

        WHILE new population size is less than population_size

            Select parent1 using tournament selection
            Select parent2 using tournament selection

            Apply crossover to produce offspring1 and offspring2

            Mutate offspring1
            Mutate offspring2

            Add offspring1 to new population

            IF population is still not full
                Add offspring2 to new population

        Replace old population with new population

    Evaluate final population fitness

    Find best final individual

    Decode best individual into:
        best_modes
        best_setpoints

    Get detailed metrics for best solution

    IF verbose
        Print final optimization summary

    RETURN dictionary containing:
        best individual
        best fitness
        best A/C modes
        best setpoints
        best metrics
        fitness history
        final population
        final fitness values
        total number of evaluations
```

---

## Method: `_get_solution_metrics`

```text
METHOD _get_solution_metrics(modes, setpoints)

    Run thermal simulation using:
        modes
        setpoints
        outdoor temperatures
        initial indoor temperature

    Calculate total electricity cost

    Get detailed electricity cost breakdown

    RETURN dictionary containing:
        indoor temperatures
        cooling powers
        electrical powers
        COP values
        total energy consumption
        total cooling delivered
        total cost
        total discomfort
        number of A/C-on hours
        mean indoor temperature
        minimum indoor temperature
        maximum indoor temperature
        mean COP
        cost breakdown
```

---

## Supporting Function: `run_optimization_scenario`

```text
FUNCTION run_optimization_scenario(
    outdoor_temps,
    scenario_name,
    weight_cost,
    weight_consumption,
    weight_discomfort,
    generations,
    random_seed,
    thermal_props
)

    Print scenario name and objective weights

    Create GAConfig using:
        cost weight
        consumption weight
        discomfort weight
        number of generations

    Create GeneticAlgorithmAC object using:
        outdoor temperatures
        config
        thermal properties

    Run optimization using:
        random seed
        verbose output enabled

    Extract best solution metrics

    Print:
        total cost
        energy consumed
        cooling delivered
        mean COP
        discomfort
        A/C on hours
        indoor temperature range

    RETURN optimization results
```

---

## Algorithm Summary

```text
BEGIN

    Load outdoor temperature data

    Configure genetic algorithm parameters

    Initialize random population of A/C schedules

    REPEAT for each generation

        Evaluate each schedule using thermal simulation

        Rank schedules by fitness

        Preserve best schedules through elitism

        Select parents using tournament selection

        Generate offspring using uniform crossover

        Mutate offspring:
            flip A/C mode genes
            perturb temperature setpoint genes

        Form new population

    END REPEAT

    Evaluate final population

    Select best schedule

    Decode best schedule into:
        A/C modes
        temperature setpoints

    Calculate detailed performance metrics

    RETURN best schedule and metrics

END
```
