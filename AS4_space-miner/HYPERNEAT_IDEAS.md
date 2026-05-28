# HyperNEAT Ideas for Space Miner

This note collects possible directions for training a Space Miner bot with HyperNEAT. The goal is to evolve a controller that gets the highest score possible while keeping the experiments reproducible and easy to compare.

## Why HyperNEAT Fits

Space Miner is a spatial control problem. Minerals, asteroids, ship heading, object velocities, and danger zones all live in a 2D geometry. HyperNEAT is useful when task geometry matters because it evolves a CPPN that generates the weights of a larger neural network from node coordinates. In this project, that means we can give the controller a meaningful substrate instead of hand-tuning every input-to-output connection.

## First Refactor

Before trying HyperNEAT, split the game into a clean simulation and a visualizer.

Recommended structure:

```text
space_miner_env.py
  SpaceMinerEnv.reset(seed)
  SpaceMinerEnv.observe()
  SpaceMinerEnv.step(action)
  SpaceMinerEnv.score
  SpaceMinerEnv.done

replay.py
  Runs one trained controller with pygame rendering.

train_neat.py
  Baseline direct NEAT trainer.

train_hyperneat.py
  HyperNEAT trainer and CPPN-to-controller decoder.
```

This makes training faster because the environment can run without a Pygame window. It also makes results fair because the same simulation is used for training and replay.

Current code issue to fix first: in `miner_neat2.py`, asteroids move only during visualization. Training currently evaluates genomes against static asteroids, then displays the best genome against moving asteroids. HyperNEAT will learn the wrong task if this stays unchanged.

## Baseline Experiment

Create a normal NEAT baseline before adding HyperNEAT.

Inputs:

```text
ship fuel
ship angle
nearest mineral dx
nearest mineral dy
nearest mineral distance
nearest asteroid dx
nearest asteroid dy
nearest asteroid vx
nearest asteroid vy
nearest asteroid distance
```

Outputs:

```text
turn_left
turn_right
thrust
mine
```

This baseline gives you a comparison point. If HyperNEAT does not beat it, the substrate or fitness function probably needs work.

## HyperNEAT Substrate Options

### Option 1: Radial Sensors

Use 16 or 32 rays around the ship. Each ray reports mineral attraction, asteroid danger, and asteroid closing speed.

Possible input layout:

```text
for each ray:
  mineral_signal
  asteroid_signal
  closing_speed_signal

global:
  fuel
  current_speed
```

Why this is a good first HyperNEAT attempt:

- The geometry is simple: sensor nodes are arranged around a circle.
- It is compact enough to train quickly.
- The substrate has obvious symmetry, which CPPNs can exploit.

### Option 2: Ship-Centered Grid

Use a `7x7` or `9x9` grid centered on the ship. Each cell encodes nearby objects.

Possible channels:

```text
mineral intensity
asteroid danger
asteroid velocity x
asteroid velocity y
```

Why this is interesting:

- It is the most natural spatial representation.
- HyperNEAT can generate regular visual-like connectivity over the grid.
- It may generalize better if the number of objects changes.

Tradeoff: more input nodes means slower training and more care needed in the decoder.

### Option 3: Object Slots

Use the nearest `k` minerals and nearest `k` asteroids as structured inputs.

Example:

```text
mineral_1: dx, dy, distance
mineral_2: dx, dy, distance
asteroid_1: dx, dy, vx, vy, distance
asteroid_2: dx, dy, vx, vy, distance
```

Why this is useful:

- Easy to implement.
- Easy to compare against the current NEAT script.

Tradeoff: it is less "geometric" than radial sensors or a grid, so HyperNEAT may have less advantage.

## CPPN Decoder Design

The CPPN is the genotype. The generated controller network is the phenotype.

For each candidate connection between substrate nodes, query the CPPN:

```text
CPPN inputs:
  x1, y1
  x2, y2
  distance
  bias

CPPN outputs:
  weight
  optional enabled flag
```

Basic rule:

```text
if abs(weight) > threshold:
    create connection with scaled weight
else:
    omit connection
```

Start with one hidden layer in the substrate:

```text
input sensor plane -> hidden plane -> output action plane
```

Later experiments can add more hidden layers or try ES-HyperNEAT, where the substrate can be inferred more automatically.

## Action Encoding

Recommended first output set:

```text
turn_left
turn_right
thrust
mine
```

Action logic:

```text
turn = turn_right - turn_left
if thrust > 0.5: accelerate
if mine > 0.5: mine
```

Alternative output set:

```text
turn_amount in [-1, 1]
thrust in [0, 1]
mine in [0, 1]
```

The discrete version is usually easier to debug. The continuous version may produce smoother behavior after the basics work.

## Fitness Function Ideas

Avoid rewarding survival too strongly by itself, or the bot may learn to coast forever without mining.

Candidate fitness:

```text
fitness =
    100.0 * minerals_collected
  +   0.05 * frames_alive
  +   0.20 * progress_toward_nearest_mineral
  -  50.0 * collision_penalty
  -   0.05 * fuel_used
```

Useful additions:

- Reward successful mining immediately.
- Reward reducing distance to the nearest mineral.
- Penalize collision and running out of fuel.
- Penalize repeated no-op behavior if the bot learns to stall.
- Evaluate each genome on multiple seeds and use the average score.
- Track worst-case score so the final bot is not just lucky.

## Experiment Plan

1. Build `SpaceMinerEnv` as a deterministic, headless environment.
2. Fix asteroid motion so training and visualization use the same dynamics.
3. Add fixed seed sets for training and evaluation.
4. Train a direct NEAT baseline.
5. Implement radial-sensor HyperNEAT.
6. Compare baseline NEAT vs HyperNEAT on the same evaluation seeds.
7. Try a grid substrate after radial sensors work.
8. Save winners and replay them with Pygame.

Recommended metrics:

```text
best_score
average_score
median_score
worst_score
minerals_collected
frames_alive
collisions
fuel_remaining
```

## Practical Implementation Options

### Build on `neat-python`

This is the most natural path for the current project. Use `neat-python` to evolve the CPPN, then write a custom decoder that turns each CPPN genome into a controller network. This keeps dependencies small and helps you understand the algorithm.

### Try `pureples`

`pureples` is a pure-Python library that contains HyperNEAT and ES-HyperNEAT implementations and depends on `neat-python`. It could be useful as reference code or as a starting point, but check compatibility before depending on it heavily.

### Try TensorNEAT Later

TensorNEAT is a newer JAX-based library with GPU-accelerated NEAT and HyperNEAT support. It is interesting if training becomes too slow, but it is a bigger jump in dependencies and project complexity.

## Resources

- [A Hypercube-Based Encoding for Evolving Large-Scale Neural Networks](https://pubmed.ncbi.nlm.nih.gov/19199382/) - the original HyperNEAT paper by Stanley, D'Ambrosio, and Gauci. Useful for understanding CPPNs, substrates, and why geometry matters.
- [An Enhanced Hypercube-Based Encoding for Evolving the Placement, Density, and Connectivity of Neurons](https://direct.mit.edu/artl/article/18/4/331/2720/An-Enhanced-Hypercube-Based-Encoding-for-Evolving) - ES-HyperNEAT paper by Risi and Stanley. Useful after basic HyperNEAT works.
- [NEAT-Python documentation](https://neat-python.readthedocs.io/en/latest/) - current Python NEAT docs. Useful for configuration, reporters, population setup, and custom experiments.
- [NEAT-Python configuration essentials](https://neat-python.readthedocs.io/en/latest/config_essentials.html) - practical guide to parameters such as population size, inputs, outputs, mutation rates, and compatibility threshold.
- [Encog HyperNEAT structure notes](https://www.heatonresearch.com/encog/neat/hyperneat_structure.html) - approachable explanation of CPPNs and substrate queries.
- [pureples GitHub repository](https://github.com/ukuleleplayer/pureples) - pure-Python HyperNEAT and ES-HyperNEAT implementation built around `neat-python`.
- [py-hyperneat on PyPI](https://pypi.org/project/py-hyperneat/) - small Python HyperNEAT package. It appears old and minimally documented, so treat it as a reference rather than an obvious dependency.
- [TensorNEAT paper](https://arxiv.org/abs/2504.08339) - paper for a GPU-accelerated NEAT library that supports CPPN and HyperNEAT variants.
- [TensorNEAT GitHub repository](https://github.com/EMI-Group/tensorneat) - JAX-based implementation with HyperNEAT examples and GPU-oriented execution.

## Suggested First Milestone

The first milestone should be modest:

```text
Train a bot with radial-sensor HyperNEAT that beats the direct NEAT baseline on average score across 20 fixed evaluation seeds.
```

This gives the project a clear success condition and keeps the implementation from expanding too quickly.
