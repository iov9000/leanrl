from __future__ import annotations

import os
import pickle
from argparse import Namespace
from collections import deque
from collections.abc import Generator, Sequence
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    from leanrl.il_utils import compute_demo_phases, validate_phase_mode
except ImportError:
    from il_utils import compute_demo_phases, validate_phase_mode

DemoDict = Dict[str, Dict[str, np.ndarray]]
TensorDict = Dict[str, torch.Tensor]

# from drqv2 import DrQV2Agent, Encoder


def _augment_demo_dict_with_phase(
    demos: Dict[str, np.ndarray],
    phase_mode: str,
) -> Dict[str, np.ndarray]:
    if validate_phase_mode(phase_mode) == "none":
        return demos

    phase, next_phase = compute_demo_phases(
        demos["done"],
        traj_ids=demos.get("traj_ids"),
    )
    demos["phase"] = phase.astype(np.float32)
    demos["next_phase"] = next_phase.astype(np.float32)

    if "support_done" in demos:
        support_phase, support_next_phase = compute_demo_phases(
            demos["support_done"],
            traj_ids=demos.get("support_traj_ids"),
        )
        demos["support_phase"] = support_phase.astype(np.float32)
        demos["support_next_phase"] = support_next_phase.astype(np.float32)

    return demos


def load_hf_demos(
    args: Namespace,
    n_demos: int,
    perturb: float = 0.0,
    bary_enhance: bool = False,
    load_support: bool = False,
) -> DemoDict:
    folder = args.demo_dir
    env_name = args.env_id
    subsample = args.subsample
    phase_mode = getattr(args, "imitation_phase_mode", "none")
    if folder is None:
        folder = "demos"
    expert_demos = {}
    if perturb > 0:
        fname = f"demo_hf_{env_name}_{n_demos}_{args.perturbation_parameters}_{0.2}_{args.demo_eps}.pkl"
    else:
        fname = f"demo_hf_{env_name}_{n_demos}.pkl"
    try:
        # expert_demos['all'] = np.load(os.path.join(folder, f"demo_hf_{env_name}_{n_demos}.npy"))
        expert_demos["all"] = pickle.load(open(os.path.join(folder, fname), "rb"))

        fn = "all"
        if load_support:
            expert_demos[fn]["support_obs"] = expert_demos[fn]["support_obs"][
                ::subsample
            ]
            expert_demos[fn]["support_acs"] = expert_demos[fn]["support_acs"][
                ::subsample
            ]
            expert_demos[fn]["support_rew"] = expert_demos[fn]["support_rew"][
                ::subsample
            ]
            expert_demos[fn]["support_done"] = expert_demos[fn]["support_done"][
                ::subsample
            ]
        else:
            expert_demos[fn]["support_obs"] = expert_demos[fn]["obs"][::subsample]
            expert_demos[fn]["support_acs"] = expert_demos[fn]["acs"][::subsample]
            expert_demos[fn]["support_rew"] = expert_demos[fn]["rew"][::subsample]
            expert_demos[fn]["support_done"] = expert_demos[fn]["done"][::subsample]

        # overwrite with subsampled after assigning to support items
        expert_demos["all"]["obs"] = expert_demos["all"]["obs"][::subsample]
        expert_demos["all"]["acs"] = expert_demos["all"]["acs"][::subsample]
        expert_demos["all"]["rew"] = expert_demos["all"]["rew"][::subsample]
        expert_demos["all"]["done"] = expert_demos["all"]["done"][::subsample]
        _augment_demo_dict_with_phase(expert_demos["all"], phase_mode)

    except:
        print("Generate demos from HuggingFace Hub first")
        assert False

    return expert_demos


def load_hf_demos_name(
    args: Namespace, fnames: Sequence[str], load_support: bool = False
) -> DemoDict:
    folder = args.demo_dir
    env_name = args.env_id
    subsample = args.subsample
    phase_mode = getattr(args, "imitation_phase_mode", "none")
    if folder is None:
        folder = "demos"
    expert_demos = {}

    for fn in fnames:
        try:
            # expert_demos['all'] = np.load(os.path.join(folder, f"demo_hf_{env_name}_{n_demos}.npy"))
            expert_demos[fn] = pickle.load(open(os.path.join(folder, fn), "rb"))
            expert_demos[fn]["obs"] = expert_demos[fn]["obs"][::subsample]
            expert_demos[fn]["acs"] = expert_demos[fn]["acs"][::subsample]
            expert_demos[fn]["rew"] = expert_demos[fn]["rew"][::subsample]
            expert_demos[fn]["done"] = expert_demos[fn]["done"][::subsample]
            if load_support:
                expert_demos[fn]["support_obs"] = expert_demos[fn]["support_obs"][
                    ::subsample
                ]
                expert_demos[fn]["support_acs"] = expert_demos[fn]["support_acs"][
                    ::subsample
                ]
                expert_demos[fn]["support_rew"] = expert_demos[fn]["support_rew"][
                    ::subsample
                ]
                expert_demos[fn]["support_done"] = expert_demos[fn]["support_done"][
                    ::subsample
                ]
            else:
                expert_demos[fn]["support_obs"] = expert_demos[fn]["obs"][::subsample]
                expert_demos[fn]["support_acs"] = expert_demos[fn]["acs"][::subsample]
                expert_demos[fn]["support_rew"] = expert_demos[fn]["rew"][::subsample]
                expert_demos[fn]["support_done"] = expert_demos[fn]["done"][::subsample]
            _augment_demo_dict_with_phase(expert_demos[fn], phase_mode)
        except:
            print("Generate demos from HuggingFace Hub first")
            assert False

    obs_all = []
    acs_all = []
    rew_all = []
    done_all = []
    support_obs_all = []
    support_acs_all = []
    support_rew_all = []
    support_done_all = []

    for k, v in expert_demos.items():
        obs_all.append(v["obs"])
        acs_all.append(v["acs"])
        rew_all.append(v["rew"])
        done_all.append(v["done"])
        support_obs_all.append(v["support_obs"])
        support_acs_all.append(v["support_acs"])
        support_rew_all.append(v["support_rew"])
        support_done_all.append(v["support_done"])

    expert_demos["all"] = {}
    expert_demos["all"]["obs"] = np.concatenate(obs_all, 0)
    expert_demos["all"]["acs"] = np.concatenate(acs_all, 0)
    expert_demos["all"]["rew"] = np.concatenate(rew_all, 0)
    expert_demos["all"]["done"] = np.concatenate(done_all, 0)
    expert_demos["all"]["support_obs"] = np.concatenate(support_obs_all, 0)
    expert_demos["all"]["support_acs"] = np.concatenate(support_acs_all, 0)
    expert_demos["all"]["support_rew"] = np.concatenate(support_rew_all, 0)
    expert_demos["all"]["support_done"] = np.concatenate(support_done_all, 0)
    _augment_demo_dict_with_phase(expert_demos["all"], phase_mode)

    return expert_demos


def load_fast_demos(
    args: Namespace,
    filename: str,
    n_demos: int = 1,
    bary_enhance: bool = False,
    offset: int = 100,
    obs_shape: int = 3,
    load_support: bool = False,
) -> DemoDict:
    expert_demos = {}
    folder = args.demo_dir
    subsample = args.subsample
    phase_mode = getattr(args, "imitation_phase_mode", "none")

    if folder is None:
        folder = "demos"
    expert_demos = {}
    expert_demos["all"] = pickle.load(open(os.path.join(folder, filename), "rb"))

    try:
        # expert_demos['all'] = np.load(os.path.join(folder, f"demo_hf_{env_name}_{n_demos}.npy"))
        expert_demos["all"] = pickle.load(open(os.path.join(folder, filename), "rb"))
        expert_demos["all"]["obs"] = expert_demos["all"]["obs"][10::subsample]
        expert_demos["all"]["acs"] = expert_demos["all"]["acs"][10::subsample]
        expert_demos["all"]["rew"] = expert_demos["all"]["rew"][10::subsample]
        expert_demos["all"]["done"] = expert_demos["all"]["done"][10::subsample]
        _augment_demo_dict_with_phase(expert_demos["all"], phase_mode)
    except:
        print(f"No FAST demo found for {filename}, go find it!")
        assert False

    return {"all": expert_demos}


def load_lap_demos(
    folder: str = "demos/davos_data",
    n_demos: int = 1,
    bary_enhance: bool = False,
    offset: int = 500,
    obs_shape: int = 3,
    filter_by_score: bool = False,
    flip_z: bool = False,
    load_support: bool = False,
) -> DemoDict:
    expert_demos = {}
    obs = []
    obs_full = []
    acs = []
    rew = []
    dones = []
    cnt = 0

    filter_name_list = []
    if filter_by_score:
        df1 = pd.read_excel(
            os.path.join(
                folder,
                "DiagnosticLaparoscopy_UnknownPathology_2021-18-10--14-48-56.xls",
            )
        )
        df2 = pd.read_excel(
            os.path.join(
                folder,
                "DiagnosticLaparoscopy_UnknownPathology_2021-18-10--14-48-57.xls",
            )
        )

        filter_name_list.extend(
            [item[:8] for item in df1[df1["Total score"] >= 145]["Attachments:"]]
        )
        filter_name_list.extend(
            [item[:8] for item in df2[df2["Total score"] >= 145]["Attachments:"]]
        )

    filtered_list = []
    for f in os.listdir(folder):
        if len(filter_name_list) == 0:
            filter_flag = True
        else:
            filter_flag = f[:8] in filter_name_list

        if "npy" in f and filter_flag:
            filtered_list.append(f)

    print("Training on best trajectories:", filtered_list)

    for f in filtered_list:
        cnt += 1
        if "npy" in f:
            demo = {}
            o = np.load(os.path.join(folder, f))
            o_tr = o[offset:-offset, :obs_shape]
            o_full = o[offset:-offset, :]
            if flip_z:
                o_tr[:, 2] = -o_tr[:, 2]
                o_full[:, 2] = -o_full[:, 2]

            a = np.zeros([len(o_tr)])
            r = np.zeros([len(o_tr)])
            d = np.zeros([len(o_tr)])
            d[-1] = 1

            obs.append(o_tr)
            obs_full.append(o_full)
            acs.append(a)
            rew.append(r)
            dones.append(d)
        if cnt > n_demos:
            break

    expert_demos["obs"] = np.concatenate(obs, 0)
    expert_demos["obs_full"] = np.concatenate(obs_full, 0)
    expert_demos["acs"] = np.concatenate(acs, 0)
    expert_demos["rew"] = np.concatenate(rew, 0)
    expert_demos["done"] = np.concatenate(dones, 0)

    return {"all": expert_demos}


def get_trajectory_list(
    demos: Mapping[str, np.ndarray], oa_cat: bool = False
) -> List[np.ndarray]:
    done_cnt = 0
    trajs = []
    ep = []
    obs = demos["obs"]
    acs = demos["acs"]
    dones = demos["done"]
    if len(acs.shape) == 1:
        acs = np.expand_dims(acs, -1)

    for o, a, d in zip(obs, acs, dones):
        if oa_cat:
            oa = np.concatenate([o, a])
        else:
            oa = o

        ep.append(oa.astype(np.float32))
        if int(d) == 1:
            done_cnt += 1
            trajs.append(np.array(ep))
            ep = []

    return trajs


def gaussian_kld(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1)


def compute_sw_barycenter(
    a: np.ndarray,
    b: np.ndarray,
    lr: float = 1e3,
    nb_iter_max: int = 1000,
    bary_size: int = 1000,
    n_proj: int = 100,
) -> np.ndarray:
    x1_torch = torch.from_numpy(a)
    x2_torch = torch.from_numpy(b)
    xbinit = np.random.randn(bary_size, a.shape[-1]).astype(np.float32)
    xbary_torch = torch.tensor(xbinit).requires_grad_(True)

    # x_all = np.zeros((nb_iter_max, xbary_torch.shape[0], 2))

    loss_iter = []

    # generator for random permutations
    gen = torch.Generator()
    gen.manual_seed(42)

    alpha = 0.5  # adaptive alpha schedule?

    for i in tqdm(range(nb_iter_max)):
        loss = alpha * ot.sliced_wasserstein_distance(
            xbary_torch, x2_torch, n_projections=n_proj, seed=gen
        ) + (1 - alpha) * ot.sliced_wasserstein_distance(
            xbary_torch, x1_torch, n_projections=n_proj, seed=gen
        )

        loss_iter.append(loss.clone().detach().cpu().numpy())
        loss.backward()

        # performs a step of projected gradient descent
        with torch.no_grad():
            grad = xbary_torch.grad
            xbary_torch -= grad * lr  # / (1 + i / 5e1)  # step
            xbary_torch.grad.zero_()
            # x_all[i, :, :] = xbary_torch.clone().detach().cpu().numpy()

    xb = xbary_torch.clone().detach().cpu().numpy()

    return xb


def enhance_demos_with_eps_balls(args: Namespace, demos: DemoDict) -> List[np.ndarray]:
    add_obs_list = []
    test_env = gym.make(args.env_id)
    for obs in demos["all"]["obs"]:
        init_obs = test_env.reset(args.seed, initial_state=obs)
        for i in range(args.eps_ball_size):
            o, r, t, tr, i = test_env.step(test_env.action_space.sample())
            add_obs_list.append(o)

    return add_obs_list


def enhance_demos_with_barycenters(
    args: Namespace,
    demos: DemoDict,
    test_env: gym.Env,
    lr: float = 1e3,
    nb_iter_max: int = 1000,
    bary_size: int = 1000,
    n_proj: int = 100,
) -> DemoDict:
    demo_list = get_trajectory_list(demos["all"], args.use_actions)
    pairs = [(a, b) for idx, a in enumerate(demo_list) for b in demo_list[idx + 1 :]]
    barys = []
    for i, (a, b) in enumerate(pairs):
        if i == args.enhance_barys:
            break
        else:
            barys.append(
                compute_sw_barycenter(a, b, lr, nb_iter_max, bary_size, n_proj)
            )

    f = plt.figure(figsize=(10, 10))
    for b in barys:
        ob_shape = test_env.observation_space.shape[-1]
        if len(test_env.action_space.shape) == 0:
            ac_shape = 1
        else:
            ac_shape = test_env.action_space.shape[-1]
        o = b[:, :ob_shape]
        a = b[:, ob_shape : ob_shape + ac_shape]
        d = b[:, -1]
        plt.scatter(o[:, 0], o[:, 1])
        if hasattr(test_env, "get_reward"):
            r = test_env.get_reward(o)
        else:
            r = np.zeros(d.shape)

        # truncate actions
        a_ = np.squeeze(a).astype(np.int32)
        a_ = np.minimum(np.maximum(0, a_), 4, a_)

        demos["all"]["obs"] = np.concatenate([demos["all"]["obs"], o], 0)
        demos["all"]["acs"] = np.concatenate([demos["all"]["acs"], a_], 0)
        demos["all"]["done"] = np.concatenate([demos["all"]["done"], d], 0)
        demos["all"]["rew"] = np.concatenate([demos["all"]["rew"], r], 0)

    plt.show()

    # demo_list.extend(barys)

    return demos


def demos_gen(
    data: Sequence[np.ndarray], batch_size: int
) -> Generator[Sequence[np.ndarray], None, None]:
    """Yields batch of specified size"""
    if batch_size <= 0:
        return
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def demos_gen_dict(
    data: Mapping[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
) -> Generator[TensorDict, None, None]:
    """Yields batch of specified size"""
    if batch_size <= 0:
        return

    b_inds = torch.arange(len(data["obs"]), device=data["obs"].device, dtype=torch.long)
    if shuffle:
        b_inds = torch.randperm(len(data["obs"]), device=data["obs"].device)

    for i in range(0, len(data["obs"]), batch_size):
        end = i + batch_size
        if drop_last and end > len(data["obs"]):
            break
        mb_inds = b_inds[i:end]
        batch = {
            "obs": data["obs"][mb_inds],
            "acs": data["acs"][mb_inds],
            "rew": data["rew"][mb_inds],
            "done": data["done"][mb_inds],
            "support_obs": data["support_obs"][mb_inds],
            "support_acs": data["support_acs"][mb_inds],
            "support_rew": data["support_rew"][mb_inds],
            "support_done": data["support_done"][mb_inds],
        }
        if "phase" in data:
            batch["phase"] = data["phase"][mb_inds]
            batch["next_phase"] = data["next_phase"][mb_inds]
        if "support_phase" in data:
            batch["support_phase"] = data["support_phase"][mb_inds]
            batch["support_next_phase"] = data["support_next_phase"][mb_inds]
        yield batch


def demos_sample_batch(
    data: Mapping[str, np.ndarray], batch_size: int, shuffle: bool = False
) -> Optional[Dict[str, np.ndarray]]:
    """Yields batch of specified size"""
    if batch_size <= 0:
        return

    i = np.random.randint(0, len(data["obs"]) - batch_size)
    end = i + batch_size
    batch = {
        "obs": data["obs"][i:end],
        "acs": data["acs"][i:end],
        "rew": data["rew"][i:end],
        "done": data["done"][i:end],
    }
    if "phase" in data:
        batch["phase"] = data["phase"][i:end]
        batch["next_phase"] = data["next_phase"][i:end]
    return batch


def get_concat_samples(
    policy_batch: Any, expert_batch: Mapping[str, np.ndarray], device: torch.device
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    policy_obs = policy_batch.observations
    policy_acs = policy_batch.actions
    policy_obs_next = policy_batch.next_observations
    policy_rew = torch.squeeze(policy_batch.rewards)
    policy_dones = torch.squeeze(policy_batch.dones)

    if isinstance(expert_batch, dict):
        expert_rew = (
            torch.from_numpy(expert_batch["rew"])
            .type(torch.get_default_dtype())
            .to(device)
        )
        expert_obs = (
            torch.from_numpy(expert_batch["obs"])
            .type(torch.get_default_dtype())
            .to(device)
        )
        expert_acs = (
            torch.from_numpy(expert_batch["acs"])
            .type(torch.get_default_dtype())
            .to(device)
        )
        expert_dones = (
            torch.from_numpy(np.squeeze(expert_batch["done"]))
            .type(torch.get_default_dtype())
            .to(device)
        )

    expert_obs_next = np.concatenate(
        [expert_batch["obs"][1:], np.expand_dims(expert_batch["obs"][-1], 0)], axis=0
    )  # repeat last observation
    expert_obs_next = (
        torch.from_numpy(expert_obs_next).type(torch.get_default_dtype()).to(device)
    )

    batch_state = torch.cat([policy_obs, expert_obs], dim=0)
    batch_next_state = torch.cat([policy_obs_next, expert_obs_next], dim=0)
    batch_action = torch.cat([policy_acs, expert_acs], dim=0)
    batch_reward = torch.cat([policy_rew, expert_rew], dim=0)
    batch_done = torch.cat([policy_dones, expert_dones], dim=0)
    is_expert = torch.cat(
        [
            torch.zeros_like(policy_rew, dtype=torch.bool),
            torch.ones_like(expert_rew, dtype=torch.bool),
        ],
        dim=0,
    )

    return (
        batch_state,
        batch_next_state,
        batch_action,
        batch_reward,
        batch_done,
        is_expert,
    )


def prepare_batch_update_irl(
    env: gym.Env,
    opt: Namespace,
    expert_demos: Mapping[str, np.ndarray] | np.ndarray,
    obs: np.ndarray | torch.Tensor,
    next_obs: np.ndarray | torch.Tensor | None,
    acs: np.ndarray | torch.Tensor,
    dones: np.ndarray | torch.Tensor,
    policy: nn.Module,
    phase: np.ndarray | torch.Tensor | None = None,
    next_phase: np.ndarray | torch.Tensor | None = None,
    compute_lprobs: bool = False,
    load_support: bool = False,
) -> TensorDict:
    ac_sample = env.single_action_space.sample()

    if isinstance(ac_sample, int) or isinstance(ac_sample, np.int64):
        ac_shape = 1
    else:
        ac_shape = ac_sample.shape[-1]

    # choose device based on policy/obs
    if isinstance(obs, torch.Tensor):
        device = obs.device
    else:
        try:
            device = next(policy.parameters()).device  # type: ignore[arg-type]
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_tensor(x, dtype=torch.get_default_dtype(), squeeze=False):
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        if dtype is not None:
            t = t.to(dtype=dtype)
        t = t.to(device)
        if squeeze:
            t = torch.squeeze(t)
        return t

    phase_mode = validate_phase_mode(getattr(opt, "imitation_phase_mode", "none"))

    obs_t = to_tensor(obs)
    next_obs_t = to_tensor(next_obs) if next_obs is not None else None
    acs_t = to_tensor(acs)
    dones_t = to_tensor(dones, squeeze=True)
    phase_t = to_tensor(phase, squeeze=True) if phase is not None else None
    next_phase_t = to_tensor(next_phase, squeeze=True) if next_phase is not None else None

    # flatten first dimension to use samples from all env
    if "atari" in opt.exp_name:
        obs_t = obs_t.reshape(-1, *obs_t.shape[2:])
        if next_obs_t is not None:
            next_obs_t = next_obs_t.reshape(-1, *next_obs_t.shape[2:])
        acs_t = acs_t.reshape(-1, ac_shape)
        dones_t = dones_t.reshape(-1)
        if phase_t is not None:
            phase_t = phase_t.reshape(-1)
        if next_phase_t is not None:
            next_phase_t = next_phase_t.reshape(-1)
    else:
        obs_t = obs_t.reshape(-1, obs_t.shape[-1])
        if next_obs_t is not None:
            next_obs_t = next_obs_t.reshape(-1, next_obs_t.shape[-1])
        acs_t = acs_t.reshape(-1, ac_shape)
        if ac_shape == 1:
            acs_t = acs_t.squeeze(-1)
        dones_t = dones_t.reshape(-1)
        if phase_t is not None:
            phase_t = phase_t.reshape(-1)
        if next_phase_t is not None:
            next_phase_t = next_phase_t.reshape(-1)

    obs_next_t = next_obs_t
    if obs_next_t is None:
        obs_next_t = torch.cat([obs_t[1:], obs_t[-1:].clone()], dim=0)
    if phase_t is None:
        phase_t = torch.zeros(obs_t.shape[0], device=device, dtype=torch.get_default_dtype())
    if next_phase_t is None:
        next_phase_t = phase_t.clone()

    # sample expert_demos
    if isinstance(expert_demos, dict):
        expert_obs_t = to_tensor(expert_demos["obs"])
        expert_acs_t = to_tensor(expert_demos["acs"])
        expert_dones_t = to_tensor(expert_demos["done"])
        expert_phase_t = (
            to_tensor(expert_demos["phase"], squeeze=True)
            if "phase" in expert_demos
            else torch.zeros(expert_obs_t.shape[0], device=device, dtype=torch.get_default_dtype())
        )
        expert_next_phase_t = (
            to_tensor(expert_demos["next_phase"], squeeze=True)
            if "next_phase" in expert_demos
            else expert_phase_t.clone()
        )
        expert_obs_next_t = (
            to_tensor(expert_demos["next_obs"])
            if "next_obs" in expert_demos
            else torch.cat([expert_obs_t[1:], expert_obs_t[-1:].clone()], dim=0)
        )
    else:
        expert_ob_ac_done_reward = expert_demos  # [np.random.randint(0, expert_demos.shape[0], opt.batch_size), :]
        expert_dones_t = to_tensor(expert_ob_ac_done_reward[:, -1])
        # rewards = expert_ob_ac_done_reward[:, -2]
        expert_ob_ac = expert_ob_ac_done_reward[:, :-2]
        expert_obs_t = to_tensor(expert_ob_ac[:, :-ac_shape])
        expert_acs_t = to_tensor(expert_ob_ac[:, -ac_shape:])
        expert_obs_next_t = torch.cat(
            [expert_obs_t[1:], expert_obs_t[-1:].clone()], dim=0
        )
        expert_phase_t = torch.zeros(
            expert_obs_t.shape[0], device=device, dtype=torch.get_default_dtype()
        )
        expert_next_phase_t = expert_phase_t.clone()

    if ac_shape == 1 and expert_acs_t.dim() > 1 and expert_acs_t.shape[-1] == 1:
        expert_acs_t = expert_acs_t.squeeze(-1)

    N = expert_obs_t.shape[0]
    T = 1000

    # policy_traj = np.empty((N, T, obs.shape[-1]), dtype=np.float32)
    # a_buffer = np.empty((N, T-1, ac_shape), dtype=np.float32)
    # expert_traj = np.empty((N, T, obs.shape[-1]), dtype=np.float32)

    # policy_ob_ac = np.concatenate([obs, acs], 1)
    # eval lprobs conditioned on obs, acs

    # print(expert_obs_t.shape, expert_acs_t.shape, obs_t.shape, acs_t.shape)

    # only necessary for AIRL
    if compute_lprobs:
        with torch.no_grad():
            if getattr(opt, "on_policy", False):
                if not opt.use_actions:
                    _, expert_lprobs_t, _, _ = policy.get_action_and_value(expert_obs_t)
                else:
                    _, expert_lprobs_t, _, _ = policy.get_action_and_value(
                        expert_obs_t, expert_acs_t
                    )

                _, policy_lprobs_t, _, _ = policy.get_action_and_value(obs_t, acs_t)
            else:
                _, expert_lprobs_t, _ = policy.get_action(expert_obs_t)
                _, policy_lprobs_t, _ = policy.get_action(obs_t)
    else:
        policy_lprobs_t = torch.zeros_like(acs_t)
        expert_lprobs_t = torch.zeros_like(expert_acs_t)

    policy_obs_t = obs_t
    policy_acs_t = acs_t
    policy_obs_next_t = obs_next_t
    policy_dones_t = dones_t
    policy_phase_t = phase_t
    policy_next_phase_t = next_phase_t
    all_obs_t = torch.cat([expert_obs_t, obs_t], dim=0)
    all_obs_next_t = torch.cat([expert_obs_next_t, policy_obs_next_t], dim=0)
    if expert_acs_t.shape[-1] == acs_t.shape[-1]:
        all_acs_t = torch.cat([expert_acs_t, acs_t], dim=0)
        all_lprobs_t = torch.cat([expert_lprobs_t, policy_lprobs_t]).type(
            torch.get_default_dtype()
        )
    else:
        all_lprobs_t = torch.cat(
            [torch.zeros_like(policy_lprobs_t), policy_lprobs_t]
        ).type(torch.get_default_dtype())
        all_acs_t = torch.cat([torch.zeros_like(acs_t), acs_t], dim=0)

    all_dones_t = torch.cat([expert_dones_t, policy_dones_t], dim=0)
    all_phase_t = torch.cat([expert_phase_t, policy_phase_t], dim=0)
    all_next_phase_t = torch.cat([expert_next_phase_t, policy_next_phase_t], dim=0)

    update_dict = {}
    update_dict["expert_obs"] = expert_obs_t
    update_dict["expert_obs_next"] = expert_obs_next_t
    update_dict["expert_acs"] = expert_acs_t
    update_dict["expert_lprobs"] = expert_lprobs_t
    update_dict["expert_dones"] = expert_dones_t
    update_dict["expert_phase"] = expert_phase_t
    update_dict["expert_next_phase"] = expert_next_phase_t

    if load_support:
        if isinstance(expert_demos, dict):
            support_obs_t = to_tensor(expert_demos["support_obs"])
            support_acs_t = to_tensor(expert_demos["support_acs"])
            support_dones_t = to_tensor(expert_demos["support_done"])
            support_phase_t = (
                to_tensor(expert_demos["support_phase"], squeeze=True)
                if "support_phase" in expert_demos
                else torch.zeros(
                    support_obs_t.shape[0],
                    device=device,
                    dtype=torch.get_default_dtype(),
                )
            )
            support_next_phase_t = (
                to_tensor(expert_demos["support_next_phase"], squeeze=True)
                if "support_next_phase" in expert_demos
                else support_phase_t.clone()
            )
            if (
                ac_shape == 1
                and support_acs_t.dim() > 1
                and support_acs_t.shape[-1] == 1
            ):
                support_acs_t = support_acs_t.squeeze(-1)
            support_obs_next_t = torch.cat(
                [support_obs_t[1:], support_obs_t[-1:].clone()], dim=0
            )
        else:
            # fallback: reuse expert tensors if support set is not provided as dict
            support_obs_t = expert_obs_t
            support_acs_t = expert_acs_t
            support_dones_t = expert_dones_t
            support_obs_next_t = expert_obs_next_t
            support_phase_t = expert_phase_t
            support_next_phase_t = expert_next_phase_t

        update_dict["support_obs"] = support_obs_t
        update_dict["support_obs_next"] = support_obs_next_t
        update_dict["support_acs"] = support_acs_t
        update_dict["support_dones"] = support_dones_t
        update_dict["support_phase"] = support_phase_t
        update_dict["support_next_phase"] = support_next_phase_t
    else:
        update_dict["support_obs"] = expert_obs_t
        update_dict["support_obs_next"] = expert_obs_next_t
        update_dict["support_acs"] = expert_acs_t
        # update_dict['support_lprobs'] = expert_lprobs_t
        update_dict["support_dones"] = expert_dones_t
        update_dict["support_phase"] = expert_phase_t
        update_dict["support_next_phase"] = expert_next_phase_t

    update_dict["policy_obs"] = policy_obs_t
    update_dict["policy_obs_next"] = policy_obs_next_t
    update_dict["policy_acs"] = policy_acs_t
    update_dict["policy_lprobs"] = policy_lprobs_t
    update_dict["policy_dones"] = policy_dones_t
    update_dict["policy_phase"] = policy_phase_t
    update_dict["policy_next_phase"] = policy_next_phase_t

    update_dict["all_obs"] = all_obs_t
    update_dict["all_obs_next"] = all_obs_next_t
    update_dict["all_acs"] = all_acs_t
    update_dict["all_lprobs"] = all_lprobs_t
    update_dict["all_dones"] = all_dones_t
    update_dict["all_phase"] = all_phase_t
    update_dict["all_next_phase"] = all_next_phase_t

    return update_dict


class ResNetAIRLDisc(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layer_blocks=2,
        hid_dim=100,
        hid_act="relu",
        use_bn=False,
        clamp_magnitude=10.0,
    ):
        super().__init__()

        if hid_act == "relu":
            hid_act_class = nn.ReLU
        elif hid_act == "tanh":
            hid_act_class = nn.Tanh
        else:
            raise NotImplementedError()

        self.clamp_magnitude = clamp_magnitude
        self.input_dim = input_dim

        self.first_fc = nn.Linear(input_dim, hid_dim)

        self.blocks_list = nn.ModuleList()
        for i in range(num_layer_blocks - 1):
            block = nn.ModuleList()
            block.append(nn.Linear(hid_dim, hid_dim))
            if use_bn:
                block.append(nn.BatchNorm1d(hid_dim))
            block.append(hid_act_class())
            self.blocks_list.append(nn.Sequential(*block))
        self.blocks_list = self.blocks_list

        self.last_fc = nn.Linear(hid_dim, 1)

    def forward(self, batch):
        x = self.first_fc(batch)
        for block in self.blocks_list:
            x = x + block(x)
        output = self.last_fc(x)
        if self.clamp_magnitude is not None:
            output = torch.clamp(
                output, min=-1.0 * self.clamp_magnitude, max=self.clamp_magnitude
            )
        return output


class AntCFWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.env.observation_space = gym.spaces.Box(
            low=-np.inf * np.ones((27,)), high=np.inf * np.ones((27,)), dtype=np.float32
        )

        super().__init__(env=self.env)

    def reset(self):
        o = self.env.reset()
        return o[:27]

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return o[:27], r, d, i


class MujocoTestWrapper(gym.Wrapper):
    def __init__(self, env, eps0=0.0, epsdyn=0.0, params=[]):
        self.env = env
        self.eps0 = eps0
        self.epsdyn = epsdyn
        self.default_params = {}
        # save default simulation parameters
        # print(dir(self.env.model))
        for p in params:
            self.default_params[p] = np.array(getattr(self.env.model, p))

        # print(dir(self.env.model))
        self.perturb_init(eps0)
        for p in params:
            self.perturb_parameter(p)

        super().__init__(env=self.env)

    def perturb_init(self, eps):
        #  perturb initial position and velocity
        self.init_qpos = np.array(self.init_qpos)
        self.init_qvel = np.array(self.init_qvel)
        self.init_qpos = self.init_qpos * eps * np.random.randn(*self.init_qpos.shape)
        # self.init_qvel = self.init_qvel * eps * \
        #    np.random.randn(*self.init_qvel.shape)
        self.init_qpos = tuple(self.init_qpos)
        self.init_qvel = tuple(self.init_qvel)

    def perturb_parameter(self, parameter="body_mass"):
        param = np.array(getattr(self.env.model, parameter))
        if parameter == "body_mass":
            param_mod = param + np.random.rand(*param.shape) * self.epsdyn
        else:
            param_mod = param + np.random.randn(*param.shape) * self.epsdyn

        # print(
        #    f">> Perturbed parameter {parameter}, default: {self.default_params[parameter]}, changed: {param_mod}")
        # assign new value
        getattr(self.env.model, parameter)[:] = param_mod

    def reset(self, seed=1, options=None):
        o = self.env.reset(seed=seed, options=options)[0]
        # reset mass to default value otherwise accum.
        for k, v in self.default_params.items():
            getattr(self.env.model, k)[:] = v

        self.perturb_init(self.eps0)
        for k, v in self.default_params.items():
            self.perturb_parameter(k)

        return o, {}

    def step(self, action):
        o, r, t, tr, i = self.env.step(action)
        d = t or tr
        # o = o + self.eps*np.random.randn(*o.shape)
        return o, r, t, tr, i


class Box2dTestWrapper(gym.Wrapper):
    def __init__(self, env, eps0=0.0, epsdyn=0.0, params=[]):
        self.env = env
        self.eps0 = eps0
        self.epsdyn = epsdyn
        self.default_params = {}
        # save default simulation parameters
        # print(dir(self.env.model))
        for p in params:
            self.default_params[p] = np.array(getattr(self.env.unwrapped, p))

        # print(dir(self.env.model))
        # self.perturb_init(eps0)
        for p in params:
            self.perturb_parameter(p)

        super().__init__(env=self.env)

    def perturb_parameter(self, parameter="body_mass"):
        param = np.array(getattr(self.env.unwrapped, parameter))
        param_mod = param + np.random.randn(*param.shape) * self.epsdyn

        if "wind" in parameter or "turbulence" in parameter:
            self.env.unwrapped.enable_wind = True

        # print(
        #    f">> Perturbed parameter {parameter}, default: {self.default_params[parameter]}, changed: {param_mod}")
        # assign new value
        setattr(self.env.unwrapped, parameter, param_mod)

    def reset(self, seed=1, options=None):
        o = self.env.reset(seed=seed, options=options)[0]
        # reset mass to default value otherwise accum.
        for k, v in self.default_params.items():
            setattr(self.env.unwrapped, k, v)

        # self.perturb_init(self.eps0)
        for k, v in self.default_params.items():
            self.perturb_parameter(k)

        # print((f"Lunar Lander Dynamics: gravity: {self.env.unwrapped.gravity}, "
        #       f"wind_power: {self.env.unwrapped.wind_power},"
        #       f"turbulence_power: {self.env.unwrapped.turbulence_power},"))

        return o, {}

    def step(self, action):
        o, r, t, tr, i = self.env.step(action)
        d = t or tr
        # o = o + self.eps*np.random.randn(*o.shape)
        return o, r, t, tr, i


def evaluate_model(
    args: Namespace,
    actor: nn.Module,
    device: torch.device,
    test_env: gym.Env | None = None,
    deterministic: bool = False,
    wrap_mujoco: bool = False,
    n_traj: int = 10,
    save_eval: Optional[str] = None,
    max_step: float = 1e7,
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    if test_env is None:
        test_env = gym.make(args.env_id)
        if wrap_mujoco:
            test_env = MujocoTestWrapper(
                test_env,
                eps0=args.test_eps,
                epsdyn=args.test_eps,
                params=args.perturbation_parameters,
            )
        if args.clip_action and isinstance(test_env.action_space, gym.spaces.Box):
            test_env = gym.wrappers.ClipAction(test_env)
        if args.normalize_obs:
            test_env = gym.wrappers.NormalizeObservation(test_env)
            test_env = gym.wrappers.TransformObservation(
                test_env, lambda obs: np.clip(obs, -10, 10)
            )

    mu, std, d = save_traj_dict(
        args,
        args.env_id,
        test_env,
        actor,
        device,
        deterministic=deterministic,
        nr_trajectories=n_traj,
        save_eval=save_eval,
        max_step=max_step,
    )

    # test_env.close()

    return mu, std, d


def save_traj_dict(
    opt: Namespace,
    env_name: str,
    env: gym.Env,
    model: Any,
    device: torch.device,
    nr_trajectories: int | Sequence[int] = 10,
    min_total_rew: float = -1e6,
    deterministic: bool = True,
    save_eval: Optional[str] = None,
    max_step: float = 1e7,
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    num_steps = 0
    expert_obs = []
    expert_acs = []
    expert_rew = []
    expert_done = []

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    print("Evaluation deterministic", deterministic)

    cnt = 0
    ep_rew = []
    while cnt < nr_trajectories:
        if opt.eval_on_different_seed:
            ob, _ = env.reset(seed=opt.seed + np.random.randint(0, 1000))
        else:
            ob, _ = env.reset(seed=opt.seed)

        done = False
        total_reward = 0
        episode_obs = []
        episode_acs = []
        episode_rew = []
        episode_done = []

        traj_len_cnt = 0
        while not done:
            with torch.no_grad():
                if "ppo" in opt.exp_name or "bc" in opt.exp_name:
                    action, logprob, _, value = model.get_action_and_value(
                        torch.Tensor(ob).to(device).unsqueeze(0), eval=deterministic
                    )
                elif "softq" in opt.exp_name:
                    action, softmax = model[0].choose_action(
                        torch.Tensor(ob).to(device).unsqueeze(0), model[1]
                    )
                elif "dqn" in opt.exp_name:
                    q_values = model(torch.Tensor(ob).to(device).unsqueeze(0))
                    action = torch.argmax(q_values, dim=1)
                else:
                    action, _, mean = model.get_action(
                        torch.Tensor(ob).to(device).unsqueeze(0)
                    )
                    if deterministic:
                        action = mean

            action = action.squeeze().detach().cpu().numpy()
            next_ob, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            ob = next_ob
            total_reward += reward
            episode_obs.append(ob)
            episode_acs.append(action)
            episode_rew.append(reward)
            episode_done.append(done)

            num_steps += 1

            traj_len_cnt += 1
            if traj_len_cnt > max_step:
                break

        ep_rew.append(total_reward)
        print("episode:", cnt, "reward:", total_reward)
        if total_reward > min_total_rew:
            cnt += 1
            expert_obs.extend(episode_obs)
            expert_acs.extend(episode_acs)
            expert_rew.extend(episode_rew)
            expert_done.extend(episode_done)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_model_{env_name}_{nr_trajectories}.pkl"
    expert_obs = np.stack(expert_obs)
    expert_acs = np.stack(expert_acs)
    expert_rew = np.stack(expert_rew)
    expert_done = np.stack(expert_done)

    # expert_obs = expert_obs.detach().cpu().numpy()
    # expert_acs = expert_acs.detach().cpu().numpy()
    # expert_rew = expert_rew.detach().cpu().numpy()
    # expert_done = expert_done.detach().cpu().numpy()

    ep_rew = np.array(ep_rew)

    print(
        "Mean / std over {nr_trajectories} episodes: ", np.mean(ep_rew), np.std(ep_rew)
    )

    d = {"obs": expert_obs, "acs": expert_acs, "rew": expert_rew, "done": expert_done}
    if save_eval is not None:
        pickle.dump(d, open(save_eval, "wb"))

    return np.mean(ep_rew), np.std(ep_rew), d


class AbsorbAfterDoneWrapper(gym.Wrapper):
    """Transition into absorbing state instead of episode termination.

    When the environment being wrapped returns `terminated=True` or `truncated=True`,
    we return an absorbing observation.
    This wrapper always returns `terminated=False` and `truncated=False`.

    A convenient way to add absorbing states to environments like MountainCar.
    """

    def __init__(
        self, env: gym.Env, absorb_reward: float = 0.0, n_absorbing_states: int = 1
    ):
        """Initialize AbsorbAfterDoneWrapper.

        Args:
          env: The wrapped Env.
          absorb_reward: The reward returned at the absorb state.
          absorb_obs: The observation returned at the absorb state. If None, then
            repeat the final observation before absorb.
        """
        super().__init__(env)
        self.absorb_reward = absorb_reward
        self.absorb_obs_default = np.zeros(self.observation_space.shape)
        self.absorb_obs_this_episode = None
        self.at_absorb_state = None
        self.n_absorbing_states = n_absorbing_states
        self.cnt = 0

    def reset(self, *args, **kwargs):
        """Reset the environment."""
        self.at_absorb_state = False
        self.absorb_obs_this_episode = None
        self.cnt = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        """Advance the environment by one step.

        This wrapped `step()` always returns terminated=False and truncated=False.

        After the first time either terminated or truncated is returned by the
        underlying Env, we enter an artificial absorb state.

        In this artificial absorb state, we stop calling
        `self.env.step(action)` (i.e. the `action` argument is entirely ignored) and
        we return fixed values for obs, rew, terminated, truncated, and info.
        The values of `obs` and `rew` depend on initialization arguments.
        `info` is always an empty dictionary.
        """
        if not self.at_absorb_state:
            obs, rew, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                # Initialize the artificial absorb state, which we will repeatedly use
                # starting on the next call to `step()`.
                self.at_absorb_state = True

                if self.absorb_obs_default is None:
                    self.absorb_obs_this_episode = obs
                else:
                    self.absorb_obs_this_episode = self.absorb_obs_default
        else:
            assert self.absorb_obs_this_episode is not None
            assert self.absorb_reward is not None
            obs = self.absorb_obs_this_episode
            rew = self.absorb_reward
            info = {}
            self.cnt += 1

        if self.cnt < self.n_absorbing_states + 1:
            return obs, rew, False, False, info
        else:
            return obs, rew, True, True, info


class FrankaObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        # gym.utils.RecordConstructorArgs.__init__(self, filter_keys=filter_keys)
        gym.ObservationWrapper.__init__(self, env)

        # print(env.observation_space['observation'])
        self.env.observation_space = env.observation_space["observation"]

    def observation(self, obs):
        obs_ = obs["observation"]
        cond_obs_1 = obs["achieved_goal"]
        cond_obs_2 = obs["desired_goal"]

        v_ = []
        for k, v in cond_obs_1.items():
            v_.append(v)
        # for k,v in cond_obs_2.items():
        #    v_.append(v)

        v_ = np.concatenate(v_, -1)

        # return np.concatenate((obs_, v_),-1)
        return obs_


class AdroitResetWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, demos):
        super().__init__(env)
        self.demos = demos

    def reset(self, *args, **kwargs):
        """Reset the environment."""

        obs, info = self.env.reset(*args, **kwargs)
        self.env.set_env_state(self.demos["all"]["init_state_dict"][0])
        return obs, info


def drq_weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class DmcObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, device, enc_out_dim=100, filename=None):
        # gym.utils.RecordConstructorArgs.__init__(self, filter_keys=filter_keys)
        gym.ObservationWrapper.__init__(self, env)

        if filename is not None:
            model = torch.load(filename, map_location=device)

        self.env.observation_space = gym.spaces.Box(
            low=-np.inf * np.ones((enc_out_dim,)),
            high=np.inf * np.ones((enc_out_dim,)),
            dtype=np.float32,
        )

    def observation(self, obs):
        with torch.no_grad():
            enc_obs = self.enc(obs)

        return enc_obs


class DemoStateSeqWrapper(gym.Wrapper):
    def __init__(self, env, opt, demos):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)
        self.use_actions = opt.use_actions
        self.n_demos = opt.n_demos
        self.use_actions = opt.use_actions
        self.episode_return = 0
        self.demos = demos
        self.obs = None
        self.curr_tgt = 0
        self.epsilon = 1e-3
        obs_space = self.env.observation_space
        # stack shapes
        self.env.observation_space = type(obs_space)(
            shape=np.tile(obs_space.low, 2).shape,
            low=np.tile(obs_space.low, 2),
            high=np.tile(obs_space.high, 2),
        )

    def reset(self, seed=1, options=None):
        obs, info = self.env.reset(seed, options)
        exp_obs_0 = self.demos["obs"][0]
        o = np.concatenate([obs, exp_obs_0], -1)

        print("Previously reached target: ", self.curr_tgt)
        self.curr_tgt = 0

        return o, info

    def compute_distance(self, obs, tgt):
        # TODO: add other distances! Currently only L2
        # Extend to learnable distances?
        return np.sqrt(np.sum((obs - tgt) ** 2))

    def step(self, action):
        # also, here, WANT state before applying action
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        done = term or trunc

        info["gt_reward"] = gt_reward
        # reset the pwil because the atoms are exhausted
        reward = -self.compute_distance(next_obs, self.demos["obs"][self.curr_tgt])

        next_obs = np.concatenate([next_obs, self.demos["obs"][self.curr_tgt]], -1)
        self.obs = next_obs

        # update sequential target if reached
        if np.abs(reward) < self.epsilon:
            self.curr_tgt += 1
        self.episode_return += reward
        if done:
            if "episode" not in info.keys():
                info["episode"] = {}
            info["episode"]["ep_rew_irl"] = self.episode_return
            self.episode_return = 0

        return next_obs, reward, term, trunc, info


class ResetToStateWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env=env)
        # self.reward_fn = make_network(**reward_fn_spec)

    def reset(self, seed=1, options=None, state=None):
        obs, info = self.env.reset(seed, options)

        print(obs)

        if state is not None:
            self.env.unwrapped.set_state(state)

        o, r, t, tr, i = self.env.step(self.env.action_space.sample() * 0.0)
        print(o)

        return o, info


class FailureBufferWrapper(gym.Wrapper):
    def __init__(self, env, maxlen=500, weight=0.1):
        super().__init__(env=env)
        self.buffer = deque(maxlen=maxlen)
        self.weight = weight
        # self.reward_fn = make_network(**reward_fn_spec)

    def compute_distance(self, obs, tgt):
        return np.sqrt(np.sum((obs - tgt) ** 2))

    def reset(self, seed=1, options=None, state=None):
        obs, info = self.env.reset()
        self.obs = obs
        self.episode_return = 0
        # print("Length of failure buffer:", len(self.buffer))

        return obs, info

    def step(self, action):
        next_obs, gt_reward, term, trunc, info = self.env.step(action)
        done = term or trunc

        if term:
            self.buffer.append(self.obs)

        # reset the pwil because the atoms are exhausted
        if len(self.buffer) > 0:
            reward = self.compute_distance(next_obs, np.array(self.buffer))
        else:
            reward = 0
        info["failstate_reward"] = reward

        self.episode_return += reward
        if done:
            if "episode" not in info.keys():
                info["episode"] = {}
            info["episode"]["ep_rew_fail"] = self.episode_return
            self.episode_return = 0

        return next_obs, gt_reward + self.weight * reward, term, trunc, info


def make_dcm(cfg: Any) -> gym.Env:
    import dmc2gym

    """Helper function to create dm_control environment"""
    if cfg.env.name == "dmc_ball_in_cup_catch":
        domain_name = "ball_in_cup"
        task_name = "catch"
    elif cfg.env.name == "dmc_point_mass_easy":
        domain_name = "point_mass"
        task_name = "easy"
    else:
        domain_name = cfg.env.name.split("_")[1]
        task_name = "_".join(cfg.env.name.split("_")[2:])

    if cfg.env.from_pixels:
        # Set env variables for Mujoco rendering
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]

        # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
        camera_id = 2 if domain_name == "quadruped" else 0

        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=cfg.seed,
            visualize_reward=False,
            from_pixels=True,
            height=cfg.env.image_size,
            width=cfg.env.image_size,
            frame_skip=cfg.env.action_repeat,
            camera_id=camera_id,
        )

        print(env.observation_space.dtype)
        # env = FrameStack(env, k=cfg.env.frame_stack)
        env = FrameStackEager(env, k=cfg.env.frame_stack)

    else:
        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=cfg.seed,
            visualize_reward=True,
        )
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env
