import argparse
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *

from Envs.CsAmpEnv import CsAmpEnv

def model(input_vec, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = input_vec
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=128,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def model_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        return t >= num_timesteps

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (num_iterations / 5, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env,
        q_func=model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=50000,
        batch_size=32,
        gamma=0.99,
        learning_starts=100,
        learning_freq=4,
        target_update_freq=500,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i) 
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(netlist, target_specs):

    env = CsAmpEnv(num_process=1 , design_netlist=netlist, target_specs=target_specs)
    return env

def main():
    max_timesteps = 10000
    # Run training
    dsn_netlist = './netlists/cs_amp.cir'
    target_spec = dict(gain_min=3.5, bw_min=1e9)
    env = CsAmpEnv(num_process=1 ,
                   design_netlist=dsn_netlist,
                   target_specs=target_spec)

    session = get_session()
    model_learn(env, session, num_timesteps=max_timesteps)

if __name__ == "__main__":
    main()