from dqn_utils import sample_n_unique
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

class ReplayBuffer2(object):
    def __init__(self, size):
        """This is a memory efficient implementation of the replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.size = size

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs_t      = None
        self.action   = None
        self.reward   = None
        self.obs_tp    = None
        self.done     = None
        self.specs    = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)

        obs_t_batch = self.obs_t[idxes]
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        obs_tp_batch = self.obs_tp[idxes]
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        specs_batch = self.specs[idxes]

        return obs_t_batch, act_batch, rew_batch, obs_tp_batch, done_mask, specs_batch


    def store_effect(self, idx, s_t, act, rew, s_tp, done, specs):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.obs_t[idx] = s_t
        self.action[idx] = s_tp
        self.reward[idx] = act
        self.obs_tp[idx] = rew
        self.done[idx] = done
        self.specs[idx] = specs


        if self.num_in_buffer < self.size:
            self.num_in_buffer += 1

        self.next_idx += 1
        return self.next_idx

if __name__ == '__main__':

    file = './cs_amp.yaml'
    env = Env(file)
    n_actions = env.n_actions

    t_max=3
    replay_buffer_size=100
    batch_size=32
    experience_rate=2

    # construct the replay buffer
    replay_buffer = ReplayBuffer2(replay_buffer_size)
    index = replay_buffer.next_idx
    last_obs = env.reset()

for t in itertools.count():

    if t > t_max:
        break
    #print (t)
    action = np.random.randint(n_actions)
    results = env.step([action])
    for result in results:
        s_t, s_tp, action, reward, done, specs = result
        print ('st:{}' .format(s_t))
        print ('a:{}' .format(action))
        print ('stp:{}' .format(s_tp))
        print ('reward:{}' .format(reward))
        print ('done:{}' .format(done))
        print ('specs:{}' .format(specs))
        # index = replay_buffer.store_effect(index, s_t, action, reward, s_tp, done, specs)




