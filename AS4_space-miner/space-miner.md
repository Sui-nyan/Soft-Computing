# Space Miner Training

This project trains an autonomous miner for the Space Miner game with NEAT neuroevolution. The miner is implemented in `miner_neat2.py` and configured by `neat_config.txt`.

## NEAT Miner

The miner evolves a feed-forward neural network with `neat-python`. Each genome represents one candidate controller. During a generation, every genome is converted into a neural network and run in the same game simulation. The simulation starts with the ship in the center of an `800 x 600` wraparound world, 5 minerals, and 8 asteroids. The genome controls the ship until it collides with an asteroid, runs out of fuel, reaches a no-mineral terminal condition, or survives for 5000 frames.

After all genomes in a generation are evaluated, NEAT keeps the best-performing structures, groups similar networks into species, applies mutation and crossover, and repeats this process for `GENERATIONS = 10`. The configured population size is 200, so each generation compares 200 candidate controllers. The best genome is saved as `winner.pkl`.

The NEAT network is configured with 5 inputs, 3 hidden nodes, and 3 outputs. It starts as a fully connected feed-forward network and can mutate connection weights, biases, enabled connections, and topology.

## Inputs

The miner receives a compact state vector:

| Input | Meaning | Normalization |
| --- | --- | --- |
| `mineral_distance` | Distance from the ship to the closest mineral. | Divided by the maximum screen diagonal distance. |
| `mineral_relative_angle` | Angle from the ship's current heading to the closest mineral. | Divided by `pi`, giving an approximate `[-1, 1]` range. |
| `asteroid_distance` | Distance from the ship to the closest asteroid. | Divided by the maximum screen diagonal distance. |
| `asteroid_relative_angle` | Angle from the ship's current heading to the closest asteroid. | Divided by `pi`. |
| `fuel` | Remaining ship fuel. | Divided by 100. |

Relative positions use wraparound distance, so the network sees the nearest direction across screen edges rather than being confused by objects near the opposite border.

## Outputs

The network produces 3 outputs:

| Output | Use |
| --- | --- |
| `turn` | Converted to a steering command with `(output[0] * 2 - 1) * 0.1`, allowing small left or right turns. |
| `thrust` | If `output[1] > 0.5`, the ship moves forward along its current heading. |
| `mine` | If `output[2] > 0.5`, the ship attempts to mine nearby minerals. |

This keeps the action space simple: the evolved network learns when to steer, when to spend fuel moving, and when to mine.

## Fitness

The miner's fitness is calculated in `calculate_fitness` as:

```text
fitness =
    minerals_collected * minerals_weight
  + alive_time * alive_time_weight
  + mineral_progress * mineral_progress_weight
  + fuel_efficiency * fuel_efficiency_weight
  - idle_time * idle_penalty
  - asteroid_collision_penalty_term
  - cumulative_steering * steering_penalty
  - asteroid_proximity * asteroid_proximity_penalty
```

where:

```text
fuel_efficiency = minerals_collected / max(fuel_used, 1)
asteroid_collision_penalty_term =
    alive_time * asteroid_collision_penalty if a collision happened, otherwise 0
```

The weights are loaded from the `[FitnessWeights]` section of `neat_config.txt`:

| Parameter | Current value | What it configures | Justification |
| --- | ---: | --- | --- |
| `minerals` | `30.0` | Reward per collected mineral. | This is the main objective, so it should dominate the fitness. A miner that collects resources is better than one that only survives or moves smoothly. |
| `alive_time` | `0.001` | Small reward per frame survived. | Survival matters because the ship needs time to reach minerals, but the value is deliberately small so the agent does not learn to avoid risk forever without mining. |
| `mineral_progress` | `0.05` | Reward for reducing the best-known distance to the current closest mineral. | This gives partial credit before minerals are collected. It helps early generations learn movement toward goals even when they rarely reach and mine a mineral. |
| `idle_penalty` | `0.0035` | Penalty per frame with no movement. | This discourages controllers that sit still to conserve fuel or avoid asteroids. It pushes exploration and active mining behavior. |
| `fuel_efficiency` | `0.03` | Reward for collected minerals per fuel used. | This encourages efficient routes and avoids rewarding wasteful thrusting. It is smaller than the mining reward because efficiency should refine behavior, not replace mining. |
| `asteroid_collision_penalty` | `0.000` | Penalty applied on collision, scaled by alive time. | This is currently disabled. The likely reason is that collisions already terminate the episode, so a separate penalty can make evolution overly conservative or punish otherwise good miners too harshly. |
| `steering_penalty` | `0.000` | Penalty for total steering magnitude. | This is currently disabled. Steering smoothness is less important than collecting minerals, and too much penalty can prevent useful course corrections. |
| `asteroid_proximity_penalty` | `0.0015` | Penalty for moving within the asteroid proximity threshold. | This gently discourages risky paths near asteroids without making the miner afraid to navigate through cluttered areas. |

The chosen weights make mineral collection the highest priority, use progress as a learning signal for incomplete attempts, and keep safety and efficiency as secondary shaping terms.
