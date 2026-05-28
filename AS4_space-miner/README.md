# Space Miner

Small Pygame project with a manual game mode and a NEAT training script.

## Notes

- [HyperNEAT ideas and resources](HYPERNEAT_IDEAS.md)

## Conda Setup

Create the environment from this folder's environment file:

```powershell
conda env create -f AS4_space-miner/environment.yml
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
python AS4_space-miner/miner.py
```

Fixed-seed harness:

```powershell
python AS4_space-miner/miner_harness.py
```

NEAT training:

```powershell
python AS4_space-miner/miner_neat2.py
```
