import sys
import os
import gymnasium as gym
# import pybullet
#import gym
import numpy as np
from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO, SAC, TD3, DQN
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy
from itertools import count

import argparse

import pickle

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# from irl.utils import AntCFWrapper

def save_expert_traj(env_name, env, model, nr_trajectories=10, min_total_rew=100):
    num_steps = 0
    expert_traj = []

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    cnt = 0
    while cnt < nr_trajectories:
        ob = env.reset()[0]
        done = False
        total_reward = 0
        episode_traj = []

        while not done:
            ac, _states = model.predict(ob)
            next_ob, reward, term, trunc, _ = env.step(ac)
            done = term or trunc
            ob = next_ob
            total_reward += reward
            if env_name == 'Ant-v3':
                stacked_vec = np.hstack([np.squeeze(ob[:27]), np.squeeze(ac), reward, done])
            else:
                stacked_vec = np.hstack([np.squeeze(ob), np.squeeze(ac), reward, done])
            episode_traj.append(stacked_vec)
            num_steps += 1

        print("episode:", cnt, "reward:", total_reward)
        if total_reward > min_total_rew:
            cnt += 1
            expert_traj.extend(episode_traj)

    filename = f"demo_hf_{env_name}_{nr_trajectories}.npy" 
    expert_traj = np.stack(expert_traj)
    np.save(filename, expert_traj)

def save_expert_traj_dict(opt, env_name, env, model, nr_trajectories=10, min_total_rew=100, extraflag=''):
    num_steps = 0
    expert_obs = []
    expert_acs = []
    expert_rew = []
    expert_done = []

    if isinstance(nr_trajectories, list):
        nr_trajectories = nr_trajectories[0]

    cnt = 0
    while cnt < nr_trajectories:
        ob = env.reset()[0]
        done = False
        total_reward = 0
        episode_obs = []
        episode_acs = []
        episode_rew = []
        episode_done = []
        coinflip = np.random.rand() > 0.5
        if opt.action_noise > 0 and coinflip:
            noise = opt.action_noise
        else:
            noise = 0.0

        while not done:
            ac, _states = model.predict(ob)
            ac = ac + np.random.randn(*ac.shape)*noise
            next_ob, reward, term, trunc, _ = env.step(ac)
            done = term or trunc
            
            ob = next_ob
            total_reward += reward
            episode_obs.append(ob)
            episode_acs.append(ac)
            episode_rew.append(reward)
            episode_done.append(done)

            num_steps += 1

        print("episode:", cnt, "reward:", total_reward)
        if total_reward > min_total_rew and total_reward < args.max_rew:
            cnt += 1
            expert_obs.extend(episode_obs)
            expert_acs.extend(episode_acs)
            expert_rew.extend(episode_rew)
            expert_done.extend(episode_done)

    filename = f"demo_hf_{env_name}_{nr_trajectories}{extraflag}.pkl" 
    expert_obs = np.stack(expert_obs)
    expert_acs = np.stack(expert_acs)
    expert_rew = np.stack(expert_rew)
    expert_done = np.stack(expert_done)

    d = {'obs': expert_obs, 'acs': expert_acs, 'rew': expert_rew, 'done': expert_done}
    pickle.dump(d, open(filename, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v5",
                        help="the name of the environment")
    parser.add_argument("--algo", type=str, default="TQC",
                        help="algorithm")
    parser.add_argument("--nr_traj", default=1, type=int,
                        required=False, help="nr of trajectories to save")
    parser.add_argument("--min_rew", default=0, type=float,
                        required=False, help="minimum cum. rew. to save traj.")
    parser.add_argument("--max_rew", default=1000000, type=float,
                        required=False, help="minimum cum. rew. to save traj.")
    parser.add_argument("--eps0", default=0.0, type=float,
                        required=False, help="initial state perturbation")
    parser.add_argument("--epsdyn", default=0.0, type=float,
                        required=False, help="dynamics perturbation")
    parser.add_argument("--action_noise", default=0.0, type=float,
                        required=False, help="action perturbation")
    parser.add_argument("--perturb_parameter", default='body_mass', type=str, nargs='+',
                        required=False, help="specify which sim parameter to perturb")
    parser.add_argument('--record_video', default=False, required=False,
                        action='store_true', help='record video')
    parser.add_argument('--normalize_obs', default=False,
                        action='store_true', required=False, help='normalize obs')
    parser.add_argument("--extraflag", type=str, default="", help="extraflag for name")

    args = parser.parse_args()

    if args.algo == 'TQC':
        checkpoint = load_from_hub(
	    repo_id=f"farama-minari/{args.env_name}-{args.algo.upper()}-expert",
	    filename=f"{args.env_name.lower()}-{args.algo}-expert.zip",
    )
    else:
        checkpoint = load_from_hub(
	    repo_id=f"farama-minari/{args.env_name}-{args.algo.upper()}-expert",
	    filename=f"{args.env_name.lower()}-{args.algo.lower()}-expert.zip",
    )
    custom_objects = {
      "learning_rate": 0.0,
      "lr_schedule": lambda _: 0.0,
      "clip_range": lambda _: 0.0,
    }
    algo = args.algo.lower()
    if algo == 'sac':
        model = SAC.load(checkpoint, custom_objects=custom_objects)
    elif algo == 'ppo':
        model = PPO.load(checkpoint, custom_objects=custom_objects)
    elif algo == 'dqn':
        model = DQN.load(checkpoint, custom_objects=custom_objects)
    elif algo == 'td3':
        model = TD3.load(checkpoint, custom_objects=custom_objects)
    elif algo == 'tqc':
        model = TQC.load(checkpoint, custom_objects=custom_objects)

    env = gym.make(args.env_name, render_mode='rgb_array')
    if args.normalize_obs:
        env = gym.wrappers.NormalizeObservation(env)
    if args.record_video:
        env = gym.wrappers.RecordVideo(env, 
            f"demo_vids/test_video_{args.env_name}_{args.algo}_initnoise_{args.eps0}__{args.perturb_parameter}_{args.epsdyn}_action_noise{args.action_noise}_{args.min_rew}_{args.max_rew}",
            episode_trigger = lambda x: x % 1 == 0)
    save_expert_traj_dict(args, args.env_name, env, model, args.nr_traj, args.min_rew, args.extraflag)


    

    

