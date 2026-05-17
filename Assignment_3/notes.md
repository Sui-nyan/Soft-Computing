CWA-25B43C74-7235-4D31-ABB9-5814DE74BB88

## Recommended project scope

**Project title:**
**Optimising Daily Air Conditioner Settings Using Weather Forecasts and Evolutionary Algorithms**

**Core idea:**
Create a simple optimiser that recommends hourly A/C settings for one day, using local weather forecast data, while balancing:

1. electricity cost,
2. energy consumption,
3. thermal discomfort.

To keep it feasible in one week, model the day as **24 hourly decisions** rather than a continuous real-time controller.

## Simplified optimisation formulation

**Decision variables**

For each hour `t = 1,...,24`, optimise:

* A/C setpoint temperature, for example 22–28°C.
* A/C mode, for example ON/OFF or cooling intensity level.
* Optional: fan level, low/medium/high.

A simple chromosome or particle could be:

`[setpoint_1, setpoint_2, ..., setpoint_24]`

or, slightly richer:

`[(setpoint_1, mode_1), ..., (setpoint_24, mode_24)]`

**Objective function**

Use a weighted sum:

`fitness = α × energy_cost + β × energy_consumption + γ × discomfort`

Where:

* **energy_consumption** is estimated from a simplified A/C power model.
* **energy_cost** = kWh × electricity price.
* **discomfort** = penalty when indoor temperature is outside a comfort range, for example 22–26°C.

This is enough to satisfy the mathematical modelling requirement without making the project too large.

## Suggested algorithm

Use a **Genetic Algorithm**, because it maps naturally to a 24-hour schedule:

* One chromosome = one full-day A/C schedule.
* Genes = hourly setpoints or mode choices.
* Fitness = cost + energy + discomfort.
* Mutation = randomly adjust one or more hourly settings.
* Crossover = combine parts of two daily schedules.

PSO would also work, but GA is easier to explain for discrete hourly schedules and easier to implement quickly.

## Data plan

A simple assumption could be:

* Small/medium room A/C: around 0.7–1.2 kW when actively cooling.
* Use a duty-cycle approximation, for example higher outdoor temperature and lower setpoint means the A/C runs more often.
* Make this explicit as a limitation.


## Suggested report outline

### Chapter 1: Problem Definition and Modeling

* Real-world context: reducing A/C energy use while maintaining comfort.
* Decision variables: hourly setpoints and/or ON/OFF state.
* Objective function: weighted sum of cost, consumption, and discomfort.
* Constraints:

  * setpoint bounds, for example 22–28°C;
  * comfort range, for example 22–26°C;
  * one setting per hour;
  * no historical consumption data, so estimated consumption model is used.
* Search space:

  * 24-hour schedule;
  * continuous or mixed discrete-continuous;
  * non-linear because weather, comfort, and cooling demand interact.

### Chapter 2: Algorithm Selection and Design

* Choose Genetic Algorithm.
* Explain why gradient-based methods are not ideal:

  * discrete choices,
  * non-linear penalties,
  * approximate simulation model.
* Map chromosome to A/C schedule.
* Define GA parameters:

  * population size: 30–50,
  * generations: 50–100,
  * mutation rate: 5–15%,
  * crossover rate: 70–90%.

### Chapter 3: Implementation and Methodology

* Pseudocode.
* Data pipeline:

  * city → weather API → hourly temperature → optimisation model.
* Constraint handling:

  * clip invalid setpoints to allowed range;
  * add penalties for discomfort.
* Termination:

  * maximum generations;
  * optional stop if fitness does not improve.

### Chapter 4: Results and Performance Analysis

Include at least:

* convergence curve;
* optimised schedule;
* baseline comparison;
* estimated daily kWh and cost;
* discomfort score;
* sensitivity analysis.

The assignment specifically asks for convergence curves, sensitivity analysis, and optionally a comparative baseline, so these should be prioritised. 

### Chapter 5: Conclusion and Practical Implications

* Was the optimised schedule practical?
* How much estimated energy/cost reduction was achieved?
* What are the limitations?

  * no measured indoor temperature data;
  * no real A/C telemetry;
  * simplified room thermal model;
  * static or simplified electricity pricing;
  * weather forecast uncertainty.
* Discuss “No Free Lunch”: GA worked for this model, but another optimiser might perform better on a different building, climate, or objective function.

## Minimum viable implementation

To avoid over-scoping, I recommend this version:

**Inputs**

* City coordinates.
* 24-hour outdoor temperature forecast.
* Assumed A/C power rating.
* Electricity price.
* Comfort temperature range.

**Outputs**

* Best hourly setpoint schedule.
* Estimated kWh.
* Estimated cost.
* Discomfort score.
* Convergence plot.
* Baseline comparison.

Climate Data
https://opendata.cwa.gov.tw/dataset/climate/C-B0024-002

Air conditioner consumption data
https://acsize.net/power-consumption-calculator/

Price data
每月用電度數分段 | --- | 夏 月 (6 月 1 日至 9 月 30 日) | 非 夏 月 (夏月以外時間) |
| ------------|-------|------|---------|
120 度以下部分 | 每 度 | 1.78 | 1.78 |
121~330 度部分 | 每 度 | 2.55 | 2.26 |
331~500 度部分 | 每 度 | 3.80 | 3.13 |
501~700 度部分 | 每 度 | 5.14 | 4.24 |
701~1000 度部分 | 每 度 | 6.44 | 5.27 |
1001 度以上部分 | 每 度 | 8.86 | 7.03 |

---

Person 1 (Elisa):
- Code Implementation for Genetic algorithm

Person 2 (Tina):
- Data visualisation + README.md

Person 3 (Jessie):
- Report Write Up