import subprocess
import os
import sys
import shlex
import tyro
from dataclasses import dataclass


@dataclass
class Args:
    test_mode: bool = False
    """If true, runs with minimal timesteps for verification."""
    base_dir: str = "."
    """Base directory containing the scripts."""
    demo_dir: str = "demos"
    """Directory to save demos."""
    compile: bool = True
    """Whether to use torch.compile."""
    cudagraphs: bool = True
    """Whether to use cudagraphs."""


def run_command(cmd):
    print(f"Running: {cmd}")
    res = subprocess.run(shlex.split(cmd), shell=False, check=True)
    return res


def main(args: Args):
    envs = ["Ant-v5", "Humanoid-v5", "HalfCheetah-v5", "Walker2d-v5", "Hopper-v5"]

    total_timesteps = 6000 if args.test_mode else 1_000_000
    eval_episodes = 1 if args.test_mode else 10

    # Ensure demo directory exists
    os.makedirs(args.demo_dir, exist_ok=True)

    for env_id in envs:
        print(f"\nProcessing {env_id}...")

        # 1. Train the agent and generate demos

        # Output filename convention: demo_hf_{env_id}_{n_demos}.pkl
        demo_out = os.path.join(args.demo_dir, f"demo_hf_{env_id}_{eval_episodes}.pkl")

        train_cmd = (
            f"python sac_continuous_action_torchcompile.py "
            f"--env-id {env_id} "
            f"--total-timesteps {total_timesteps} "
            f"--cuda "
            f"{'--compile ' if args.compile else ''}"
            f"{'--cudagraphs ' if args.cudagraphs else ''}"
            f"--save-interval {total_timesteps} "  # Save at the end
            f"--save-dir checkpoints/{env_id} "
            f"--save-demo "
            f"--demo-out {demo_out} "
            f"--eval-episodes {eval_episodes} "
        )
        run_command(train_cmd)

        print(f"Finished {env_id}. Demo saved to {demo_out}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
