# Project name

**Optimising Daily Air Conditioner Settings Using Weather Forecasts and Evolutionary Algorithms**

## Project Description

This project uses a Genetic Algorithm(GA) to optimise air conditioning operation schedules and setpoint temperatures over a continuous 672-hour (28-day) horizon, based on known hourly outdoor temperature data.

The objective is to maintain indoor thermal comfort while finding an optimal trade-off among electricity cost, energy consumption, and thermal comfort.


## Features

- **Long-term scheduling optimisation**: Generates a complete 28-day air conditioning control strategy based on hourly temperature data.

- **Multi-objective optimisation framework**: 

    Simultaneously balances three objectives:

  1. Electricity cost (using a tiered pricing model from Taipower)
  2. Energy consumption (using a temperature-dependent variable-speed AC power model)
  3. Thermal comfort (penalty for deviations outside the comfort range)

- **Scenario-based analysis**:

    Supports comparison of three strategies: `Comfort-First`, `Cost-Efficient`, and `Consumption-Minimizing`.

- **Genetic Algorithm optimisation**: Uses a population-based search method to perform global optimisation, effectively handling a high-dimensional problem (672 decision variables).
- **Visualisation analysis**: Produces convergence curves for each scenario, as well as dual-axis time-series plots of temperature and air-conditioning settings.

## File Structure
* `algorithm.ipynb`: Main program for running simulations and configuring weight parameters.
* `genetic_algorithm.py`: Contains the core logic of the Genetic Algorithm and the fitness function.
* `thermal_dynamics.py`: Room thermal dynamics simulator and air-conditioning energy consumption calculation module.

* `energy_consumption_utils.py`:📌 **same as the next ???**
* `energy_cost_utils.py`: calculates electricity costs using Taipei's tiered pricing structure.
* `diagnostic.py`: Diagnostic script to investigate the calculation errors
* `import_weather_data.py`:Automatically cleans and processes raw JSON weather data from the Central Weather Administration (CWA), extracts Taipei station data, and exports CSV files for different time periods for further analysis.

* `visualization.py`: Produces convergence curves and combined time-series plots of temperature and optimised A/C control states.


##  Inputs & Outputs

**[Inputs]**

* Hourly outdoor temperature data (28 days, 672 hours in total)

**[Outputs]**

* Optimised 28-day air conditioning schedule (`hourly_simulation_details_summer.csv`)
* Scenario analysis summary reports (total cost, energy consumption, and comfort score)
* Convergence curve data and time-series visualisation plots

## Quick start
### 1. Run the simulation

```bash
jupyter notebook algorithm.ipynb
```
### 2. Generated outputs

- Optimised A/C schedules
- Scenario analysis reports
- Convergence plots
- Temperature visualisations


## Visualisation Output

![Convergence plot](convergence_plot.png)

![Zoomed-in 7-Day Temperature and A/C State Timeline](Comfort-First_7days.png)
![Zoomed-in 7-Day Temperature and A/C State Timeline](Comfort-First_7days.png)
![Zoomed-in 7-Day Temperature and A/C State Timeline](Consumption-Minimizing_7days.png)

## Data Sources

* Climate Data :
https://opendata.cwa.gov.tw/dataset/climate/C-B0024-002

* Air conditioner consumption data :
https://acsize.net/power-consumption-calculator/

* Price data - Electricity Pricing (Taipower) :

| Monthly Usage | Summer Rate | Non-Summer Rate |
|---|---|---|
| Below 120 kWh | 1.78 | 1.78 |
| 121–330 kWh | 2.55 | 2.26 |
| 331–500 kWh | 3.80 | 3.13 |
| 501–700 kWh | 5.14 | 4.24 |
| 701–1000 kWh | 6.44 | 5.27 |
| Above 1000 kWh | 8.46 | 7.03 |

## Team Responsibilities

* Person 1 (Elisa) : Code Implementation for Genetic algorithm 
* Person 2 (Tina)  : Data visualisation + README.md            
* Person 3 (Jessie): Report Write Up                           

