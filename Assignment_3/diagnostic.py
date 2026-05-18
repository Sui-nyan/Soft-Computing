"""
Diagnostic script to investigate the calculation errors
"""

import numpy as np
from genetic_algorithm import GeneticAlgorithmAC, GAConfig
from thermal_dynamics import ThermalSimulator

# Example outdoor temperatures (typical summer day in Taipei)
outdoor_temps = np.array([
    22, 21, 20, 19, 19, 21, 24, 27, 30, 32, 34, 35,
    36, 35, 34, 33, 31, 29, 27, 25, 24, 23, 22, 21
], dtype=float)

# Create a test individual
individual = np.zeros(48)
# All A/C ON
individual[::2] = np.ones(24)
# Setpoints at 22°C (lowest)
individual[1::2] = np.full(24, 22.0)

print("Test Individual:")
print(f"Modes: {individual[::2].astype(int)}")
print(f"Setpoints: {individual[1::2]}")
print()

# Test the thermal simulation directly
config = GAConfig()
ga = GeneticAlgorithmAC(outdoor_temps, config)

# Decode the individual
modes, setpoints = ga._decode_individual(individual)
print(f"Decoded Modes: {modes}")
print(f"Decoded Setpoints: {setpoints}")
print()

# Simulate
sim_results = ga.thermal_sim.simulate_day(
    modes=modes,
    setpoints=setpoints,
    outdoor_temps=outdoor_temps,
    initial_indoor_temp=24.0
)

print("Simulation Results:")
print(f"Indoor temps: {sim_results['indoor_temps']}")
print(f"Cooling powers (W): {sim_results['cooling_powers'][:5]}... (first 5)")
print(f"Electrical powers (W): {sim_results['electrical_powers'][:5]}... (first 5)")
print(f"COP values: {sim_results['cop_values'][:5]}... (first 5)")
print(f"Total energy (kWh): {sim_results['total_energy_kwh']}")
print(f"Total cooling (kWh): {sim_results['total_cooling_kwh']}")
print(f"Total discomfort: {sim_results['total_discomfort']}")
print(f"AC on hours: {sim_results['ac_on_hours']}")
print(f"Mean COP: {sim_results['mean_cop']}")
print()

# Calculate cost
total_cost = ga.pricing.calculate_cost(sim_results['total_energy_kwh'])
print(f"Total cost: {total_cost} TWD")
print()

# Check fitness
fitness = ga._evaluate_fitness(individual)
print(f"Fitness: {fitness}")
print()

# Let's also check what happens with all A/C OFF
individual_off = np.zeros(48)
individual_off[::2] = np.zeros(24)  # All A/C OFF
individual_off[1::2] = np.full(24, 22.0)  # Setpoints don't matter when OFF

modes_off, setpoints_off = ga._decode_individual(individual_off)
sim_results_off = ga.thermal_sim.simulate_day(
    modes=modes_off,
    setpoints=setpoints_off,
    outdoor_temps=outdoor_temps,
    initial_indoor_temp=24.0
)

print("Simulation Results (A/C OFF):")
print(f"Indoor temps: {sim_results_off['indoor_temps']}")
print(f"Cooling powers (W): {sim_results_off['cooling_powers']}")
print(f"Electrical powers (W): {sim_results_off['electrical_powers']}")
print(f"Total energy (kWh): {sim_results_off['total_energy_kwh']}")
print(f"Total discomfort: {sim_results_off['total_discomfort']}")
print()

fitness_off = ga._evaluate_fitness(individual_off)
print(f"Fitness (A/C OFF): {fitness_off}")
