"""
Experiment orchestrator for VIAR paper.

Phase 1: Baselines + Attention Analysis (diagnostic)
Phase 2: VIAR method evaluation 
Phase 3: Ablations
Phase 4: Analysis for figures

Usage:
    python src/run_experiments.py --phase 1
    python src/run_experiments.py --phase 2
    python src/run_experiments.py --phase 3
"""

import subprocess
import sys
import json
import os
from pathlib import Path

MODELS = [
    "llava-hf/llava-1.5-7b-hf",
    # "llava-hf/llava-1.5-13b-hf",  # Enable if budget allows
]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_modal(experiment: str, model: str = MODELS[0], method: str = "baseline",
              extra_args: list = None):
    """Run a Modal experiment and return the result."""
    cmd = [
        "modal", "run", "src/eval_modal.py",
        "--experiment", experiment,
        "--model", model,
        "--method", method,
    ]
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def phase1_baselines():
    """Phase 1: Run all baseline evaluations + attention analysis."""
    print("\n" + "="*60)
    print("PHASE 1: Baselines + Attention Analysis")
    print("="*60)
    
    for model in MODELS:
        model_short = model.split("/")[-1]
        
        # 1. Attention analysis (diagnostic, for Figure 1)
        print(f"\n--- Attention Analysis: {model_short} ---")
        run_modal("attention_analysis", model=model)
        
        # 2. Text-only baseline (measures text prior exploitation)
        print(f"\n--- Text-Only Baselines: {model_short} ---")
        run_modal("text_only", model=model)
        
        # 3. POPE baseline
        print(f"\n--- POPE Baseline: {model_short} ---")
        run_modal("pope", model=model, method="baseline")
        
        # 4. MMStar baseline
        print(f"\n--- MMStar Baseline: {model_short} ---")
        run_modal("mmstar", model=model, method="baseline")
        
        # 5. MME baseline
        print(f"\n--- MME Baseline: {model_short} ---")
        run_modal("mme", model=model, method="baseline")


def phase2_viar():
    """Phase 2: Run VIAR method on all benchmarks."""
    print("\n" + "="*60)
    print("PHASE 2: VIAR Method Evaluation")
    print("="*60)
    
    for model in MODELS:
        model_short = model.split("/")[-1]
        
        for benchmark in ["pope", "mmstar", "mme"]:
            print(f"\n--- VIAR on {benchmark}: {model_short} ---")
            run_modal(benchmark, model=model, method="viar")


def phase3_ablations():
    """Phase 3: Ablation studies on VIAR components."""
    print("\n" + "="*60)
    print("PHASE 3: Ablation Studies")
    print("="*60)
    
    # Will be implemented after Phase 1-2 results inform which ablations matter
    print("Ablations to be run after reviewing Phase 1-2 results")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    args = parser.parse_args()
    
    if args.phase == 1:
        phase1_baselines()
    elif args.phase == 2:
        phase2_viar()
    elif args.phase == 3:
        phase3_ablations()
