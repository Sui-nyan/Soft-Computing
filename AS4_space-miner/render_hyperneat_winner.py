import argparse
import os
import pickle
import tempfile
from pathlib import Path

import HyperNEAT_v1 as hyperneat


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WINNER_PATH = SCRIPT_DIR / "hyperneat_v1_winner.pkl"


def parse_args():
    parser = argparse.ArgumentParser(description="Render a saved HyperNEAT Space Miner winner")
    parser.add_argument(
        "--winner",
        default=str(DEFAULT_WINNER_PATH),
        help="Path to the saved winner pickle produced by HyperNEAT_v1.py",
    )
    parser.add_argument("--seed", type=int, help="Override the saved render seed")
    parser.add_argument("--fps", type=int, help="Override the saved render FPS")
    parser.add_argument("--hide-grid", action="store_true", help="Hide the ship-centered sensor grid")
    return parser.parse_args()


def load_winner_bundle(path):
    with open(path, "rb") as winner_file:
        bundle = pickle.load(winner_file)

    if not isinstance(bundle, dict) or "winner" not in bundle or "experiment_config" not in bundle:
        raise ValueError(
            "Expected a winner bundle saved by HyperNEAT_v1.py with 'winner' "
            "and 'experiment_config' entries."
        )
    return bundle


def create_neat_config_from_text(config_text, neat_module):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".ini", delete=False) as temp_file:
        temp_file.write(config_text)
        temp_path = temp_file.name
    try:
        return neat_module.Config(
            neat_module.DefaultGenome,
            neat_module.DefaultReproduction,
            neat_module.DefaultSpeciesSet,
            neat_module.DefaultStagnation,
            temp_path,
        )
    finally:
        os.unlink(temp_path)


def main():
    args = parse_args()
    winner_path = Path(args.winner)
    bundle = load_winner_bundle(winner_path)

    config = bundle["experiment_config"]
    if args.seed is not None:
        config["render"]["seed"] = args.seed
    if args.fps is not None:
        config["render"]["fps"] = args.fps
    if args.hide_grid:
        config["render"]["show_grid"] = False

    neat_module = hyperneat.import_neat()
    neat_config_text = bundle.get("neat_config_text") or hyperneat.build_neat_config_text(config)
    neat_config = create_neat_config_from_text(neat_config_text, neat_module)
    winner_cppn = neat_module.nn.FeedForwardNetwork.create(bundle["winner"], neat_config)
    controller = hyperneat.HyperNEATController(winner_cppn, config)

    print(f"Rendering winner from {winner_path}")
    print(f"Seed: {int(config['render']['seed'])}")
    hyperneat.render_controller(controller, config, int(config["render"]["seed"]))


if __name__ == "__main__":
    main()
