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



class Env(object):
    def __init__(self, dsn_yaml_file):

        #params_char = dict('cload': {'range': (0,10e-12), 'res': 10e-15})

        with open(dsn_yaml_file, 'r') as f:
            block_specs = yaml.load(f)
            self.dsn_netlist = block_specs['dsn_netlist']
            self.specs = block_specs['target_specs']
            self.var_chars = block_specs['variable_characteristics']
            self._compute_action_space(self.var_chars)
            self.module = block_specs['module_name']
            self.class_name = block_specs['class_name']
            self.n_process = block_specs['number_process']


    def _compute_action_space(self, var_chars):
        # type: () -> None
        """Sets up the action space based on input variable characteristics

        Parameters
        ----------
        var_chars : dict
            variables characteristics, namely range and resolution of each action
            defined over their space. format: 'vbias': {'range': (0, 1.8), 'res': 0.005}
        Returns
        -------
            None
        """
        self.params = var_chars.keys()
        self.n_params = len(self.params)
        self.n_actions = 2 ** self.n_params

        lower_bound, upper_bound, res = [], [], []
        for dict in var_chars.values():
            lower_bound.append(dict['range'][0])
            upper_bound.append(dict['range'][1])
            res.append(dict['res'])

        self.lower_bound = np.array(lower_bound)
        self.upper_bound = np.array(upper_bound)
        self.res = np.array(res)

    def _action_to_vec(self, action):
        vec1 = (action & (1 << np.arange(self.n_params)) > 0)
        vec2 = vec1[::-1]*2-1
        return vec2

    def generate_random_state_vec(self, seed=None):
        state = []
        for param in self.params:
            np.random.seed(seed)
            lower, upper = self.var_chars[param]['range']
            step = self.var_chars[param]['res']
            values = np.arange(lower, upper, step)
            state.append(np.random.choice(values))
            if seed is not None:
                seed += 1
        return state

    def reset(self, seed=None):
        # type: (int) -> None
        """Resets the environment to a random initial state for further experiment.

        Parameters
        ----------
        seed : int
            if None, true random initialization will happen, o.w pseudo random reset
            will take place
        Returns
        -------
            None
        """
        self.cur_state_vec = self.generate_random_state_vec(seed)
        # print(self.cur_state_vec)

    def step(self, actions):
        # type: (List[int]) -> List[Tuple[np.ndarray, np.array, int, float, bool]]
        """from action returns whatever needs to be stored in replay buffer

        Parameters
        ----------
        action : int
            integer number between 0 and self.n_action representing the action
            of delta change in variables
        Returns
        -------
        cur_state: np.ndarray
            current state represented as a vector of the order by which the env percepts
        next_state: np.ndarray
            next state represented as a vector of the order by which the env percepts
        action: int
            action we took
        reward: float
            reward of the state, action pair

        """
        state_dicts = []
        for action in actions:
            action_vec = self._action_to_vec(action)
            next_state_vec = self.res * action_vec + self.cur_state_vec
            next_state_dict = dict(zip(self.params, next_state_vec))
            state_dicts.append(next_state_dict)
            self.cur_state_vec = next_state_vec
            # print(action_vec)
            # print(self.cur_state_vec)

        EnvClass = importlib.import_module(self.module, package=self.class_name)
        cs_env = EnvClass.CsAmpEnv(num_process=self.n_process,
                          design_netlist=self.dsn_netlist,
                          target_specs=self.specs)
        pprint.pprint (state_dicts)
        sim_results = cs_env.run(state_dicts)

        db = []
        for idx, result in enumerate(sim_results):
            state_tp_dict = result[0]
            reward = result[1][0]
            terminate = result[1][1]
            action = actions[idx]
            state_t_dict = state_dicts[idx]

            state_tp_vec = [state_tp_dict[key] for key in sorted(state_tp_dict.keys())]
            state_t_vec = [state_t_dict[key] for key in sorted(state_t_dict.keys())]
            db_tuple = (state_t_vec, state_tp_vec, action, reward, terminate)
            db.append(db_tuple)

        return db








if __name__ == '__main__':

    file = './cs_amp.yaml'
    env = Env(file)
    n = env.n_actions
    # action = np.random.randint(n)
    # print(env._action_to_vec(action))
    env.reset()
    actions = range(n)
    print (env.cur_state_vec)
    exper = env.step(actions)
    pprint.pprint(exper)
