# Report: Reward and Penalty Design in `train_neat_for_test_agent.py`

## Overview

`train_neat_for_test_agent.py` trains a NEAT neural-network agent for the Space Miner game. The agent controls a spaceship that must survive, collect minerals, manage fuel, and avoid moving asteroids. Each genome is evaluated by running one fixed episode and assigning it a fitness score. This fitness score is the main learning signal: rewards increase fitness for useful behavior, while penalties reduce fitness for dangerous or wasteful behavior.

The episode ends when the ship collides with an asteroid, runs out of fuel, or reaches the frame limit. The final fitness is computed in `score_episode()`.

## Base Reward

The script begins with a base test score:

```python
test_score = (alive_time / 4) + (ship.minerals * 100)
```

This gives the agent two primary objectives:

- Survive longer: every frame alive adds `0.25` fitness points.
- Collect minerals: each mined mineral adds `100` fitness points.

This base score strongly favors agents that can stay alive while collecting resources. Mineral collection is weighted much more heavily than simple survival, so the agent is encouraged to actively play the game instead of only avoiding danger.

## Mineral-Seeking Rewards

The script adds two shaping rewards to help the agent learn how to reach minerals before it reliably mines them.

### Mineral Progress Reward

```python
fitness += mineral_progress * MINERAL_PROGRESS_WEIGHT
```

`MINERAL_PROGRESS_WEIGHT` is `0.05`. The variable `mineral_progress` increases when the ship reaches a new closest distance to the current target mineral. This rewards long-term improvement toward a mineral, even if the agent does not mine it during that episode.

This is useful because mining is a sparse reward: the agent only receives the large mineral reward after physically reaching and mining a mineral. Without progress shaping, early generations might receive little guidance about which movements are useful.

### Mineral Approach Reward

```python
fitness += mineral_approach * MINERAL_APPROACH_WEIGHT
```

`MINERAL_APPROACH_WEIGHT` is `0.02`. `mineral_approach` increases whenever the ship moves closer to the nearest mineral during a frame.

This gives the agent a smaller, immediate reward for moving in the right direction. Together, mineral progress and mineral approach encourage navigation behavior such as turning toward minerals, thrusting when aligned, and continuing to close distance.

## Fuel Rewards

The agent receives two fuel-related rewards.

### Fuel Efficiency Reward

```python
fuel_efficiency = ship.minerals / max(fuel_used, 1)
fitness += fuel_efficiency * FUEL_EFFICIENCY_WEIGHT
```

`FUEL_EFFICIENCY_WEIGHT` is `25.0`. This rewards collecting minerals while using less fuel. Since moving consumes fuel, the agent is encouraged to take shorter or more direct paths instead of drifting aimlessly.

### Remaining Fuel Reward

```python
fitness += ship.fuel * 0.2
```

The agent also receives a small reward for ending the episode with fuel remaining. At full fuel, this can add up to `20` points. This reinforces careful movement and makes fuel conservation valuable even when the agent has not collected many minerals.

Mining also restores fuel in the harness, so successful mining indirectly helps survival and future movement.

## Asteroid Avoidance Penalties

The script penalizes both asteroid danger and actual collision.

### Asteroid Danger Penalty

```python
fitness -= asteroid_danger * 3.0
```

`asteroid_danger` increases when the ship comes within `ASTEROID_DANGER_MARGIN`, which is `20` pixels beyond the combined ship and asteroid radii. The closer the ship gets inside this danger margin, the larger the accumulated penalty.

This penalty teaches the agent to avoid near misses, not only actual collisions. That is important because collision alone is a delayed and severe signal; the danger penalty gives earlier feedback that a trajectory is risky.

### Asteroid Collision Penalty

```python
if death_reason == "asteroid_collision":
    fitness -= 500
```

If the ship crashes into an asteroid, the agent loses `500` fitness points. This is the largest explicit penalty in the script. It strongly discourages reckless paths and makes survival around asteroids a core part of the learned strategy.

## Fuel Failure Penalty

```python
elif death_reason == "out_of_fuel":
    fitness -= 150
```

Running out of fuel subtracts `150` points. This is less severe than an asteroid collision, but still significant. The smaller size makes sense because fuel loss is often a strategic failure rather than an immediate navigation disaster. The agent should avoid wasting fuel, but the task still prioritizes asteroid safety and mineral collection.

## Idle and Wasted Action Penalties

The script also penalizes unproductive behavior.

### Idle Penalty

```python
fitness -= idle_time * 0.02
```

`idle_time` increases whenever the agent chooses not to thrust. This penalty is small, but it discourages the agent from simply sitting still to gain survival points. Because survival already gives a reward, the idle penalty helps prevent passive strategies.

### Wasted Mining Penalty

```python
wasted_mines = max(0, mine_attempts - successful_mines)
fitness -= wasted_mines * 0.03
```

The agent has a mining output, but mining only helps when the ship overlaps a mineral. This penalty subtracts a small amount for mine attempts that do not collect anything.

The penalty is intentionally light. It discourages constant spam-mining, but it does not punish exploration too harshly while the agent is still learning when to mine.

## Overall Effect

The reward system combines a main objective with several shaping signals:

- The main goal is to collect minerals and survive.
- Navigation shaping rewards moving closer to minerals before successful mining occurs.
- Fuel rewards encourage efficient movement and resource conservation.
- Asteroid penalties teach both safe spacing and collision avoidance.
- Idle and wasted-mining penalties discourage passive or noisy behavior.

This design helps NEAT learn more smoothly because the agent receives feedback throughout the episode, not only at the end. The strongest incentives are mineral collection, survival, and avoiding asteroid collisions, while the smaller rewards and penalties guide the agent toward cleaner and more efficient behavior.
