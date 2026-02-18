# =============================================================================
# main.py — Swiss Multi-Asset Portfolio Optimizer
# Runs all 4 analytical modules in sequence
# =============================================================================

import subprocess
import sys
import time
import os

SCRIPTS = [
    ("src/part1_markowitz_smi.py",      "Part 1 — Markowitz SMI"),
    ("src/part2_black_litterman.py",    "Part 2 — Black-Litterman"),
    ("src/part3_crypto_integration.py", "Part 3 — Crypto Integration"),
    ("src/part4_walk_forward.py",       "Part 4 — Walk-Forward Backtest"),
]

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  SWISS MULTI-ASSET PORTFOLIO OPTIMIZER")
    print("  Running full pipeline...")
    print("="*65)

    total_start = time.time()

    for script, label in SCRIPTS:
        print(f"\n{'='*65}")
        print(f"  {label}")
        print(f"{'='*65}")
        start = time.time()

        result = subprocess.run(
            [sys.executable, script],
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"\n✓ {label} completed in {elapsed:.1f}s")
        else:
            print(f"\n✗ {label} FAILED after {elapsed:.1f}s")
            sys.exit(1)

    total = time.time() - total_start
    print("\n" + "="*65)
    print("  ALL PARTS COMPLETE")
    print(f"  Total time : {total:.1f}s")
    print("  Results    : reports/figures/")
    print("="*65)