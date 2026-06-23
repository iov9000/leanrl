from __future__ import annotations

import copy
from argparse import Namespace
from typing import Any, Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
import ot


def get_trajectory_list_from_dict(demos: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
    traj_obs: List[List[np.ndarray]] = []
    traj_acs: List[List[np.ndarray]] = []
    traj_rew: List[List[np.ndarray]] = []

    ep_o, ep_a, ep_r = [], [], []
    for i, done in enumerate(demos["done"]):
        ep_o.append(demos["obs"][i])
        ep_a.append(demos["acs"][i])
        ep_r.append(demos["rew"][i])
        if done == 1:
            traj_obs.append(ep_o)
            traj_acs.append(ep_a)
            traj_rew.append(ep_r)
            ep_o, ep_a, ep_r = [], [], []
    if ep_o:
        traj_obs.append(ep_o)
        traj_acs.append(ep_a)
        traj_rew.append(ep_r)

    return {"obs": traj_obs, "acs": traj_acs, "rew": traj_rew}


class PWILRewarder:
    """Rewarder class to compute PWIL rewards."""

    def __init__(
        self,
        demonstrations: Dict[str, np.ndarray],
        subsampling: int,
        env: gym.Env,
        num_demonstrations: int = 1,
        time_horizon: float = 1000.0,
        alpha: float = 5.0,
        beta: float = 5.0,
        observation_only: bool = False,
        random_offset: int = 1,
    ):
        if num_demonstrations == -1:
            self.num_demonstrations = len(demonstrations["obs"])
        else:
            self.num_demonstrations = num_demonstrations

        self.time_horizon = time_horizon
        self.subsampling = max(1, subsampling)
        self.random_offset = max(0, random_offset)

        ob_shapes = list(env.observation_space.shape)
        ac_shapes = list(env.action_space.shape)
        dim_act = ac_shapes[-1] if ac_shapes else 1
        dim_obs = ob_shapes[-1]

        self.reward_sigma = beta * time_horizon / np.sqrt(dim_act + dim_obs)
        self.reward_scale = alpha

        self.observation_only = observation_only
        self.demonstrations = self.filter_demonstrations(
            get_trajectory_list_from_dict(demonstrations)
        )
        obs_atoms = self._flatten_demonstrations(self.demonstrations["obs"], "obs")

        if self.observation_only:
            dim_demos = dim_obs
            self.vectorized_demonstrations = obs_atoms
        else:
            dim_demos = dim_obs + dim_act
            acs_atoms = self._flatten_demonstrations(self.demonstrations["acs"], "acs")
            self.vectorized_demonstrations = np.concatenate(
                [obs_atoms, acs_atoms], axis=-1
            )

        self.vectorized_demonstrations = np.reshape(
            self.vectorized_demonstrations, [-1, dim_demos]
        )
        # If subsampling/random_offset wiped everything out, retry without extra thinning
        if self.vectorized_demonstrations.shape[0] == 0:
            # fall back to no extra subsampling / offset
            self.subsampling = 1
            self.random_offset = 0
            self.demonstrations = get_trajectory_list_from_dict(demonstrations)
            obs_atoms = self._flatten_demonstrations(self.demonstrations["obs"], "obs")
            if self.observation_only:
                self.vectorized_demonstrations = obs_atoms
            else:
                acs_atoms = self._flatten_demonstrations(
                    self.demonstrations["acs"], "acs"
                )
                self.vectorized_demonstrations = np.concatenate(
                    [obs_atoms, acs_atoms], axis=-1
                )
            self.vectorized_demonstrations = np.reshape(
                self.vectorized_demonstrations, [-1, dim_demos]
            )
            if self.vectorized_demonstrations.shape[0] == 0:
                raise ValueError(
                    "No demonstration samples available for PWIL after subsampling. "
                    "Reduce --subsample/--subsampling or random_offset."
                )
        self.scaler = self.get_scaler()
        self.reset()

    def filter_demonstrations(
        self, demonstrations: Dict[str, List[np.ndarray]]
    ) -> Dict[str, List[np.ndarray]]:
        filtered = {"obs": [], "acs": [], "rew": []}
        for key, trajs in demonstrations.items():
            for episode in trajs[: self.num_demonstrations]:
                random_offset = self.random_offset
                subsampled = episode[random_offset :: self.subsampling]
                filtered[key].append(subsampled)
        return filtered

    @staticmethod
    def _flatten_demonstrations(
        episodes: Sequence[Sequence[np.ndarray]], name: str
    ) -> np.ndarray:
        arrays = [
            np.asarray(episode, dtype=np.float32)
            for episode in episodes
            if len(episode) > 0
        ]
        if not arrays:
            return np.empty((0, 0), dtype=np.float32)
        try:
            return np.concatenate(arrays, axis=0)
        except ValueError as exc:
            raise ValueError(
                f"PWIL {name} demonstrations have incompatible shapes"
            ) from exc

    def get_scaler(self):
        """Defines a scaler to derive the standardized Euclidean distance."""
        return _PWILNormalizer(self.vectorized_demonstrations)

    def reset(self) -> None:
        """Makes all expert transitions available and initialize weights."""
        self.expert_atoms = copy.deepcopy(
            self.scaler.transform(self.vectorized_demonstrations)
        )
        num_expert_atoms = len(self.expert_atoms)
        self.expert_weights = np.ones(num_expert_atoms) / num_expert_atoms

    def compute_reward(
        self, obs: np.ndarray, action: np.ndarray | int | None = None
    ) -> np.ndarray:
        """Computes reward as presented in Algorithm 1."""
        if action is None:
            agent_atom = obs
        else:
            if not isinstance(action, int):
                action = np.array([action])
            if len(obs.shape) != len(action.shape):
                obs = np.expand_dims(obs, axis=0)
            agent_atom = np.squeeze(np.concatenate([obs, action], axis=-1))

        agent_atom = np.expand_dims(agent_atom, axis=0)  # add dim for scaler
        agent_atom = self.scaler.transform(agent_atom)[0]

        cost = 0.0
        weight = 1.0 / self.time_horizon - 1e-6
        norms = np.linalg.norm(self.expert_atoms - agent_atom, axis=1)

        while weight > 0:
            argmin = norms.argmin()
            expert_weight = self.expert_weights[argmin]

            if weight >= expert_weight:
                weight -= expert_weight
                cost += expert_weight * norms[argmin]
                self.expert_weights = np.delete(self.expert_weights, argmin, 0)
                self.expert_atoms = np.delete(self.expert_atoms, argmin, 0)
                norms = np.delete(norms, argmin, 0)
            else:
                cost += weight * norms[argmin]
                self.expert_weights[argmin] -= weight
                weight = 0

        reward = self.reward_scale * np.exp(-self.reward_sigma * cost)
        return reward.astype("float32")

    def compute_w2_dist_to_expert(self, trajectory: List[np.ndarray]) -> float:
        """Computes Wasserstein 2 distance to expert demonstrations."""
        self.reset()
        if self.observation_only:
            trajectory = [t["observation"] for t in trajectory]
        else:
            trajectory = [
                np.concatenate([t["observation"], t["action"]]) for t in trajectory
            ]

        trajectory = self.scaler.transform(trajectory)
        trajectory_weights = 1.0 / len(trajectory) * np.ones(len(trajectory))
        cost_matrix = ot.dist(trajectory, self.expert_atoms, metric="euclidean")
        w2_dist = ot.emd2(trajectory_weights, self.expert_weights, cost_matrix)
        return float(w2_dist)


class _PWILNormalizer:
    """Matches Kaixhin's expert-atom normalisation while handling zero variance."""

    def __init__(self, atoms: np.ndarray) -> None:
        self.mean = atoms.mean(axis=0, keepdims=True)
        ddof = 1 if atoms.shape[0] > 1 else 0
        scale = atoms.std(axis=0, ddof=ddof, keepdims=True)
        self.scale = np.where(scale == 0, 1.0, scale)

    def transform(self, atoms: np.ndarray) -> np.ndarray:
        return (np.asarray(atoms, dtype=np.float32) - self.mean) / self.scale


class PWILReward(gym.Wrapper):
    def __init__(
        self, env: gym.Env, opt: Namespace, demos: Dict[str, np.ndarray]
    ) -> None:
        super().__init__(env=env)
        self.use_actions = getattr(opt, "use_actions", True)
        self.n_demos = getattr(opt, "n_demos", 1)
        self.subsampling = getattr(opt, "subsampling", getattr(opt, "subsample", 1))
        self.episode_return = 0.0
        self.obs = None
        time_horizon = float(getattr(opt, "pwil_time_horizon", 0.0))
        if time_horizon <= 0.0:
            env_spec = getattr(env, "spec", None)
            time_horizon = float(getattr(env_spec, "max_episode_steps", None) or 1000.0)
        self.pwil = PWILRewarder(
            demos,
            subsampling=self.subsampling,
            env=env,
            num_demonstrations=self.n_demos,
            time_horizon=time_horizon,
            observation_only=not self.use_actions,
            alpha=getattr(opt, "pwil_alpha", 5.0),
            beta=getattr(opt, "pwil_beta", 5.0),
            random_offset=getattr(opt, "random_offset", 1),
        )

    def reset(
        self, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.obs = obs
        self.episode_return = 0.0
        self.pwil.reset()
        return obs, info

    def step(
        self, action: np.ndarray | int | float
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        done = term or trunc
        obs = next_obs if self.obs is None else self.obs
        info["gt_reward"] = gt_reward

        if self.use_actions:
            reward = self.pwil.compute_reward(obs, action)
        else:
            reward = self.pwil.compute_reward(obs, action=None)

        self.obs = next_obs
        self.episode_return += reward
        if done:
            info.setdefault("episode", {})
            info["episode"]["ep_rew_irl"] = self.episode_return
            self.episode_return = 0.0
            self.pwil.reset()

        return next_obs, reward, term, trunc, info
