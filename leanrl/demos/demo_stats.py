import sys
import numpy as np
import pickle

if __name__ == '__main__':
    demo_name = sys.argv[1]
    if 'npy' in demo_name:
        demo = np.load(demo_name)
        reward = demo[:,-2]
        dones = demo[:,-1] 
    else:
        demo = pickle.load(open(demo_name,'rb'))
        reward = demo['rew']
        dones = demo['done']
    
    num_episodes = np.sum(dones)
    rew_avg = np.mean(reward)
    rew_std = np.std(reward)
    rew_min = np.min(reward)
    rew_max = np.max(reward)
    
    ep_rew_list = []
    ep_rew = 0
    if 'npy' in demo_name:
        for sard in demo:
            ep_rew += sard[-2]
            if sard[-1] == 1:
                ep_rew_list.append(ep_rew)
                print("episode_reward", ep_rew)
                ep_rew = 0
    else:
        for i in range(len(demo['obs'])):
            ep_rew += reward[i]
            if dones[i] == 1:
                ep_rew_list.append(ep_rew)
                print("episode_reward", ep_rew)
                ep_rew = 0

        
    ep_rew_avg = np.mean(ep_rew_list)
    ep_rew_std = np.std(ep_rew_list)
    ep_rew_min = np.min(ep_rew_list)
    ep_rew_max = np.max(ep_rew_list)


    print("Demo file stats")
    print(demo_name)
    print("-------------")
    print("Number of episodes: ", num_episodes)
    print("Reward stats: ", rew_avg, " +- ", rew_std) 
    print("Reward min / max", rew_min, " / ", rew_max) 
    print("Episode reward stats: ", ep_rew_avg, " +- ", ep_rew_std) 
    print("Episode reward min / max", ep_rew_min, " / ", ep_rew_max) 


