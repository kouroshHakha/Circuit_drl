from dqn_utils import *
import numpy as np
import random
import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint
import yaml
import importlib
import itertools

from Env import Env


if __name__ == '__main__':

    file = './cs_amp.yaml'
    env = Env(file)
    n_actions = env.n_actions

    t_max=1
    replay_buffer_size=8
    batch_size=3
    experience_rate=3

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)
    index = replay_buffer.next_idx
    last_obs = env.reset()
    actions = []
    epsilon = 1

    for t in itertools.count():

        if t > t_max:
            break
        #print (t)

        # print (last_obs)
        # print (env.get_transition(last_obs, 7))

        p = np.random.rand()

        if (p < epsilon):
            # do something with probability of epsilon (e.g. explore)
            # print ("random policy ... ")
            action = np.random.randint(n_actions)
        else:
            # do something with probability of 1 - epsilon (e.g. exploit)
            # print ("following policy ... ")
            # action = session.run(best_actions_target_nn, feed_dict={obs_tp1_ph: [recent_obs]})
            pass

        last_obs = env.get_transition(last_obs, action)
        actions.append(action)

        if len(actions) == experience_rate:

            results = env.step(actions)
            actions.clear()
            for result in results:
                s_t, s_tp, action, reward, done, specs = result
                print ('st:{}' .format(s_t))
                print ('a:{}' .format(action))
                print ('stp:{}' .format(s_tp))
                print ('reward:{}' .format(reward))
                print ('done:{}' .format(done))
                print ('specs:{}' .format(specs))
                if (done):
                    break
                index = replay_buffer.store_effect(index, s_t, action, reward, s_tp, done, specs)


            #print ('can sample {}'.format(replay_buffer.can_sample(batch_size)))

            if (replay_buffer.can_sample(batch_size)):
                obs_t_batch, act_batch, rew_batch, obs_tp_batch, done_mask, specs_batch = \
                    replay_buffer.sample(batch_size)

                pprint.pprint ('st:{}' .format(obs_t_batch))
                pprint.pprint ('a:{}' .format(act_batch))
                pprint.pprint ('reward:{}' .format(rew_batch))
                pprint.pprint ('stp:{}' .format(obs_tp_batch))
                pprint.pprint ('done:{}' .format(done_mask))
                pprint.pprint ('specs:{}' .format(specs_batch))



