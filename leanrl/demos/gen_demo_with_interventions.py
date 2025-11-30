import argparse
import pickle
from math import floor
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
from huggingface_sb3 import load_from_hub
from sb3_contrib import TQC
from stable_baselines3 import SAC


def load_expert_model(env_name: str, algo: str):
    """Load a frozen expert policy from the Minari HuggingFace hub."""
    algo_upper = algo.upper()
    repo_id = f"farama-minari/{env_name}-{algo_upper}-expert"
    if algo_upper == "TQC":
        filename = f"{env_name.lower()}-{algo}-expert.zip"
    else:
        filename = f"{env_name.lower()}-{algo.lower()}-expert.zip"

    checkpoint_path = load_from_hub(repo_id=repo_id, filename=filename)
    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    if algo_upper == "TQC":
        return TQC.load(checkpoint_path, custom_objects=custom_objects)
    if algo_upper == "SAC":
        return SAC.load(checkpoint_path, custom_objects=custom_objects)
    raise ValueError(f"Unsupported algorithm requested: {algo}")


def build_env(args, algo_label: str):
    env = gym.make(args.env_name, render_mode="rgb_array")
    if args.normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    if args.record_video:
        video_dir = (
            f"demo_vids/test_video_{args.env_name}_{algo_label}"
            f"_ratio_{args.tqc_ratio}_action_noise{args.action_noise}"
        )
        env = gym.wrappers.RecordVideo(
            env, video_dir, episode_trigger=lambda x: x % 1 == 0
        )
    return env


def collect_expert_traj_dict(
    args,
    model,
    algo_label: str,
    nr_trajectories: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Collect trajectory dictionaries following the structure used by gen_demo."""
    if nr_trajectories <= 0:
        return None

    env = build_env(args, algo_label)
    expert_obs: List[np.ndarray] = []
    expert_acs: List[np.ndarray] = []
    expert_rew: List[float] = []
    expert_done: List[bool] = []

    cnt = 0
    while cnt < nr_trajectories:
        ob, _ = env.reset()
        done = False
        total_reward = 0.0

        episode_obs = []
        episode_acs = []
        episode_rew = []
        episode_done = []

        noise = args.action_noise if args.action_noise > 0 and np.random.rand() > 0.5 else 0.0

        while not done:
            ac, _ = model.predict(ob)
            ac = ac + np.random.randn(*ac.shape) * noise
            next_ob, reward, term, trunc, _ = env.step(ac)
            done = bool(term or trunc)
            ob = next_ob
            total_reward += reward

            episode_obs.append(ob)
            episode_acs.append(ac)
            episode_rew.append(reward)
            episode_done.append(done)

        print(f"[{algo_label}] episode: {cnt} reward: {total_reward}")
        if total_reward > args.min_rew and total_reward < args.max_rew:
            cnt += 1
            expert_obs.extend(episode_obs)
            expert_acs.extend(episode_acs)
            expert_rew.extend(episode_rew)
            expert_done.extend(episode_done)

    env.close()
    return {
        "obs": np.stack(expert_obs),
        "acs": np.stack(expert_acs),
        "rew": np.stack(expert_rew),
        "done": np.stack(expert_done),
    }


def concatenate_datasets(datasets: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    combined: Dict[str, np.ndarray] = {}
    for key in ("obs", "acs", "rew", "done"):
        combined[key] = np.concatenate([d[key] for d in datasets if d is not None], axis=0)
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", type=str, default="HalfCheetah-v5", help="name of the environment"
    )
    parser.add_argument(
        "--nr_traj",
        default=1,
        type=int,
        help="total number of trajectories to save across both experts",
    )
    parser.add_argument(
        "--min_rew",
        default=0,
        type=float,
        help="minimum cumulative reward to keep a trajectory",
    )
    parser.add_argument(
        "--max_rew",
        default=1_000_000,
        type=float,
        help="maximum cumulative reward to keep a trajectory",
    )
    parser.add_argument(
        "--action_noise",
        default=0.0,
        type=float,
        help="standard deviation of Gaussian action noise (applied randomly)",
    )
    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="record rollouts to demo_vids/",
    )
    parser.add_argument(
        "--normalize_obs",
        default=False,
        action="store_true",
        help="normalize observations using gym wrapper",
    )
    parser.add_argument(
        "--extraflag",
        type=str,
        default="_diverse",
        help="suffix for the demo filename (always appends _diverse if missing)",
    )
    parser.add_argument(
        "--tqc_ratio",
        type=float,
        default=0.5,
        help="ratio of trajectories generated with the TQC expert (between 0 and 1)",
    )

    args = parser.parse_args()
    args.tqc_ratio = max(0.0, min(1.0, args.tqc_ratio))

    if args.nr_traj <= 0:
        raise ValueError("nr_traj must be a positive integer.")

    tqc_traj = int(floor(args.nr_traj * args.tqc_ratio + 0.5))
    sac_traj = args.nr_traj - tqc_traj

    if tqc_traj == 0 and sac_traj == 0:
        raise ValueError("No trajectories requested for either expert.")

    datasets: List[Dict[str, np.ndarray]] = []

    if tqc_traj > 0:
        tqc_model = load_expert_model(args.env_name, "TQC")
        tqc_data = collect_expert_traj_dict(args, tqc_model, "TQC", tqc_traj)
        if tqc_data:
            datasets.append(tqc_data)

    if sac_traj > 0:
        sac_model = load_expert_model(args.env_name, "SAC")
        sac_data = collect_expert_traj_dict(args, sac_model, "SAC", sac_traj)
        if sac_data:
            datasets.append(sac_data)

    if not datasets:
        raise RuntimeError("Failed to collect any trajectories from either expert.")

    combined_dataset = concatenate_datasets(datasets)

    extra_suffix = args.extraflag or ""
    if "_diverse" not in extra_suffix:
        extra_suffix = f"{extra_suffix}_diverse" if extra_suffix else "_diverse"

    filename = f"demo_hf_{args.env_name}_{args.nr_traj}{extra_suffix}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(combined_dataset, f)

    print(
        f"Saved {args.nr_traj} combined trajectories "
        f"({tqc_traj} TQC / {sac_traj} SAC) to {filename}"
    )
