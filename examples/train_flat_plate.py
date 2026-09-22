"""Generate flat-plate data and train four ELM models in memory."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import train_models


def main():
    parser = argparse.ArgumentParser(description="Generate flat-plate data and train four ELM models.")
    parser.add_argument("--variant", choices=("base", "linear", "log"), default="log")
    parser.add_argument("--neurons", type=int, default=490)
    parser.add_argument("--compact", action="store_true", help="smaller data set for a quick experiment")
    args = parser.parse_args()
    _, amp_h, amp_a, data = train_models(args.variant, args.neurons, args.compact)
    print(f"Generated {len(data.h):,} samples and trained four {args.neurons}-neuron ELMs.")
    print(f"Final amplitudes: heave={amp_h:.4g} m, pitch={amp_a:.4g} rad.")


if __name__ == "__main__":
    main()
