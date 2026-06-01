# Space Miner

Small Pygame project with a manual game mode, a fixed-seed harness, a NEAT
training script, and a first HyperNEAT experiment.

## Notes

- [HyperNEAT ideas and resources](HYPERNEAT_IDEAS.md)

## Conda Setup

Create the environment from this folder's environment file:

```powershell
conda env create -f environment.yml
conda activate space-miner
```

If you are already inside `AS4_space-miner`, use:

```powershell
conda env create -f environment.yml
conda activate space-miner
```

## Run

Manual game:

```powershell
python miner.py
```

Fixed-seed harness:

```powershell
python miner_harness.py
```

NEAT training:

```powershell
python miner_neat2.py
```

HyperNEAT v1 training with built-in defaults:

```powershell
python HyperNEAT_v1.py --no-render
```

Write a tunable JSON config:

```powershell
python HyperNEAT_v1.py --write-default-config hyperneat_v1_config.json
```

Train HyperNEAT v1 with a JSON config:

```powershell
python HyperNEAT_v1.py --config hyperneat_v1_config.json --no-render
```

Train and replay the winner:

```powershell
python HyperNEAT_v1.py --config hyperneat_v1_config.json --render-winner
```

Replay a saved HyperNEAT v1 winner without retraining:

```powershell
python render_hyperneat_winner.py
```

## HyperNEAT v1

`HyperNEAT_v1.py` evolves a CPPN with `neat-python`, then decodes it into a
fixed neural controller using geometric substrate coordinates. The controller
observes a ship-centered local occupancy grid with mineral, asteroid danger,
and asteroid velocity channels.

The default action outputs are:

```text
heading_x
heading_y
thrust
mine
```

`heading_x` and `heading_y` are converted into a desired heading with
`atan2`, which avoids the wraparound discontinuity of a single angle output.

By default, training saves:

```text
hyperneat_v1_winner.pkl
hyperneat_v1_metrics.json
```
