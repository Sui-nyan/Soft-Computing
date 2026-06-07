# Space Miner Training

This project trains an autonomous miner for the Space Miner game with NEAT neuroevolution. The miner is implemented in `miner_neat2.py` and configured by `neat_config.txt`.

## NEAT Miner

The miner evolves a feed-forward neural network with `neat-python`. Each genome represents one candidate controller. During a generation, every genome is converted into a neural network and run in the same game simulation. The simulation starts with the ship in the center of an `800 x 600` wraparound world, 5 minerals, and 8 asteroids. The genome controls the ship until it collides with an asteroid, runs out of fuel, reaches a no-mineral terminal condition, or survives for 5000 frames.

After all genomes in a generation are evaluated, NEAT keeps the best-performing structures, groups similar networks into species, applies mutation and crossover, and repeats this process for `GENERATIONS = 10`. The configured population size is 200, so each generation compares 200 candidate controllers. The best genome is saved as `winner.pkl`.

The NEAT network is configured with 13 inputs, 3 hidden nodes, and 3 outputs. It starts as a fully connected feed-forward network and can mutate connection weights, biases, enabled connections, and topology.

## Inputs

The miner receives a compact state vector built from the closest mineral, the closest asteroid, and the ship's current fuel. Relative positions use wraparound distance, so the network sees the nearest direction across screen edges rather than being confused by objects near the opposite border.

| Input | Meaning | Normalization |
| --- | --- | --- |
| `mineral_distance` | Distance from the ship to the closest mineral. | Divided by the maximum screen diagonal distance. |
| `mineral_relative_angle` | Angle from the ship's current heading to the closest mineral. | Divided by `pi`, giving an approximate `[-1, 1]` range. |
| `asteroid_distance` | Distance from the ship to the closest asteroid. | Divided by the maximum screen diagonal distance. |
| `asteroid_relative_angle` | Angle from the ship's current heading to the closest asteroid. | Divided by `pi`. |
| `fuel` | Remaining ship fuel. | Divided by 100. |
| `mineral_relative_x/y` | Wraparound relative vector from the ship to the closest mineral. | `x` divided by half screen width, `y` divided by half screen height. |
| `asteroid_relative_x/y` | Wraparound relative vector from the ship to the closest asteroid. | `x` divided by half screen width, `y` divided by half screen height. |
| `asteroid_relative_velocity_x/y` | Velocity of the closest asteroid relative to the ship, so the network can infer closing motion. | Divided by the ship speed. |
| `asteroid_in_front` | How aligned the closest asteroid is with the ship's current heading. | Dot-product signal in `[0, 1]`. |
| `asteroid_time_to_collision` | Short-horizon collision-risk signal for the closest asteroid. | `0` for no near-term threat, up to `1` for more urgent threats. |

If no mineral is available, the mineral-related inputs are filled with zeros. This keeps the neural network input length stable.

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
  + mineral_approach * mineral_approach_weight
  + mineral_heading_alignment * mineral_heading_alignment_weight
  + fuel_efficiency * fuel_efficiency_weight
  - idle_time * idle_penalty
  - mineral_retreat * mineral_retreat_penalty
  - indecision * indecision_penalty
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
| `minerals` | `35.5` | Reward per collected mineral. | This is the main objective, so it should dominate the fitness. A miner that collects resources is better than one that only survives or moves smoothly. |
| `alive_time` | `0.0001` | Small reward per frame survived. | Survival matters because the ship needs time to reach minerals, but the value is deliberately small so the agent does not learn to avoid risk forever without mining. |
| `mineral_progress` | `0.05` | Reward for reducing the best-known distance to the current closest mineral. | This gives partial credit before minerals are collected. It helps early generations learn movement toward goals even when they rarely reach and mine a mineral. |
| `mineral_approach` | `0.05` | Reward for moving closer to the current target mineral on the current frame. | This gives dense moment-to-moment feedback, making it easier for NEAT to discover useful navigation before full mineral collection becomes common. |
| `mineral_retreat_penalty` | `0.05` | Penalty for moving farther away from the current target mineral. | This separates bad movement from good movement instead of letting both collapse into one ambiguous approach value. It also keeps back-and-forth oscillation from becoming profitable. |
| `mineral_heading_alignment` | `0.0001` | Reward for pointing toward the current target mineral. | Heading is useful as a weak shaping signal, but it stays small because pointing at a mineral is less important than actually moving toward and collecting it. |
| `idle_penalty` | `0.0035` | Penalty per frame with no movement. | This discourages controllers that sit still to conserve fuel or avoid asteroids. It pushes exploration and active mining behavior. |
| `indecision_penalty` | `0.01` | Penalty for no-op behavior, spinning while stationary, and long periods without mineral progress while far from a mineral. | This targets the observed behavior where ships hesitate, drift into unproductive loops, or look visually worse even when survival is decent. |
| `fuel_efficiency` | `0.03` | Reward for collected minerals per fuel used. | This encourages efficient routes and avoids rewarding wasteful thrusting. It is smaller than the mining reward because efficiency should refine behavior, not replace mining. |
| `asteroid_collision_penalty` | `0.002` | Penalty applied on collision, scaled by alive time. | Collision already ends the episode, but a small explicit penalty helps distinguish a risky survivor from a cleaner survivor with similar mineral count. |
| `steering_penalty` | `0.000` | Penalty for total steering magnitude. | This is currently disabled. Steering smoothness is less important than collecting minerals, and too much penalty can prevent useful course corrections. |
| `asteroid_proximity_penalty` | `0.0015` | Penalty for moving within the asteroid proximity threshold. | This gently discourages risky paths near asteroids without making the miner afraid to navigate through cluttered areas. |

The chosen weights make mineral collection the highest priority, use progress as a learning signal for incomplete attempts, and keep safety and efficiency as secondary shaping terms.

## Diagnostic Overlay

During best-genome visualization, the overlay now shows both behavior values and weighted fitness components:

| Overlay value | Why it is included |
| --- | --- |
| `Alive`, `Minerals`, `Fuel`, `Fuel used` | Shows whether a genome is surviving, collecting, and spending fuel efficiently. |
| `Near mineral`, `Near asteroid` | Helps explain whether the ship is actually navigating toward resources or merely surviving near danger. |
| `Progress+`, `Approach+`, `Retreat-` | Separates useful mineral-seeking movement from movement that looks active but increases the target distance. |
| `Idle`, `Indecision-`, `Steering` | Makes hesitation, stationary spinning, and excessive control noise visible during playback. |
| `Ast danger-`, `Hit asteroid` | Shows whether asteroid avoidance is a real learned behavior or whether a genome is simply getting lucky. |
| `Fit minerals`, `Fit approach`, `Fit retreat`, `Fit idle`, `Fit indecision`, `Fit asteroid` | Shows how the raw behavior values affect the final fitness score. |

This makes sense because visual quality and fitness can disagree. A ship may look better in an early generation while receiving lower fitness because it collected fewer minerals, moved away from the target, idled too much, or survived by chance. Showing the raw values and the weighted fitness terms makes those disagreements debuggable instead of relying on visual impression alone.

## Improvement Suggestions Added

| Suggestion | Implementation | Justification |
| --- | --- | --- |
| Add a fitness diagnostic overlay. | `TrainingVisualizer.draw_stats` now displays raw counters, distances, penalties, and selected weighted fitness terms. | It explains why one genome scores better than another and helps tune weights based on evidence rather than visual guesswork. |
| Reward progress toward minerals. | `mineral_approach` now accumulates positive frame-to-frame movement toward the current closest mineral, while `mineral_progress` still tracks best-distance improvement. | Dense progress reward gives early generations a learning signal before they reliably collect minerals. |
| Penalize indecision. | `indecision` increases when the ship idles, spins while stationary, or spends too long far from minerals without making progress. | This discourages controllers that survive by doing little, jittering, or looping instead of actively mining. |
