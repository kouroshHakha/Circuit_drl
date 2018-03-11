import sys
import itertools
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *

import matplotlib.pyplot as plt
import pickle
import os

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def report(tag, tsr):
    print("[{}] {} : {}".format(tag, tsr.name, tsr.get_shape()))

def learn(env,
          q_func,
          optimizer_spec,
          session,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          parallelization_rate=1):
    """Run Deep Q-learning algorithm.


    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env:
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            in: tf.Tensor
                tensorflow tensor representing the input state
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    parallelization_rate:

    """

    ###############
    # BUILD MODEL #
    ###############

    num_actions = env.n_actions
    num_params = env.n_params

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.float32, [None] + [num_params])
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.float32, [None] + [num_params])
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    ######
    
    # YOUR CODE HERE

    q_nn = q_func(obs_t_ph, num_actions, scope="q_func", reuse=False)
    target_nn = q_func(obs_tp1_ph, num_actions, scope="target_q_func", reuse=False)

    # important trick??
    # for each transition sample get Q(s,a) of the observed action and state, Q is from Q NN
    # first translate each action to its one-hot encoding so it has one element of 1 and otherwise 0.
    # then element wise multiplication would only have the right Q(s,a) in each row.
    # sum over each row to get the right Q(s,a) as a tensor
    q_t = tf.reduce_sum(q_nn*tf.one_hot(act_t_ph, num_actions), axis=1)

    # for each transition sample get target = r + gamma * max_ap(Q(sp,ap)), Q is from target NN
    max_q_tp = tf.reduce_max(target_nn, axis=1)
    target = rew_t_ph + gamma * max_q_tp * (1 - done_mask_ph)
    best_actions_target_nn = tf.argmax(target_nn, axis=1)


    total_error = tf.reduce_mean(tf.square(q_t - target))
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    ######

    report("q_nn", q_nn)
    report("target_q_nn", target_nn)
    report("q_t", q_t)
    report("max_target_q_nn", max_q_tp)
    report("target", target)
    report("act_t_ph", act_t_ph)
    report("total_error", total_error)

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size)
    index = replay_buffer.next_idx

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 100
    actions = []

    # values for plotting
    x_list = []
    error_list = []
    gain_list, bw_list, score_list, reward_list, ibias_list = [], [], [], [], []
    last_state_list = []

    t = 0
    for k in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(t):
            break

        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced many steps, and the replay buffer should contain more
        # transitions, the number of steps depends on the the number of actions
        # passed to the step
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # env.step([actions])
        # this steps the environment forward to the number of actions
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)

        #####
        
        # YOUR CODE HERE

        # epsilon greedy algorithm
        epsilon = exploration.value(t)
        p = np.random.rand()

        if (p < epsilon or not model_initialized):
            # do something with probability of epsilon (e.g. explore)
            # print ("random policy ... ")
            action = np.random.randint(num_actions)
        else:
            # do something with probability of 1 - epsilon (e.g. exploit)
            # print ("following policy @ t = %d" %(t))
            action = session.run(best_actions_target_nn, feed_dict={obs_tp1_ph: [last_obs]})

        last_obs = env.get_transition(last_obs, action)
        actions.append(action)

        if (t % parallelization_rate == 0 and not model_initialized) or (t % learning_freq and model_initialized):
            results = env.step(actions)
            actions.clear()
            for res_idx, result in enumerate(results):
                s_t, s_tp, action, reward, done, specs = result
                if (done):
                    t = t - (len(actions) - res_idx + 1)
                    last_obs = env.reset()
                    print ("reset happened @ t=%d" %t)
                    break
                # save reward, and done in the buffer
                index = replay_buffer.store_effect(index, s_t, action, reward, s_tp, done, specs)

            #####

            # at this point, the environment should have been advanced a couple of steps
            # (and reset if done was true), and last_obs should point to the new latest
            # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).

            obs_t_batch, act_batch, rew_batch, obs_tp_batch, done_batch, specs_batch = \
                replay_buffer.sample(batch_size)

            # 3.b: initialize the model if it has not been initialized yet; to do
            # that, call
            #    initialize_interdependent_variables(session, tf.global_variables(), {
            #        obs_t_ph: obs_t_batch,
            #        obs_tp1_ph: obs_tp1_batch,
            #    })
            # where obs_t_batch and obs_tp1_batch are the batches of observations at
            # the current and next time step. The boolean variable model_initialized
            # indicates whether or not the model has been initialized.
            # Remember that you have to update the target network too (see 3.d)!
            if not model_initialized:
                print ("Initializing Network ...")
                initialize_interdependent_variables(session, tf.global_variables(),
                                                    {obs_t_ph: obs_t_batch,
                                                    obs_tp1_ph: obs_tp_batch})
                model_initialized = True

            # 3.c: train the model. To do this, you'll need to use the train_fn and
            # total_error ops that were created earlier: total_error is what you
            # created to compute the total Bellman error in a batch, and train_fn
            # will actually perform a gradient step and update the network parameters
            # to reduce total_error. When calling session.run on these you'll need to
            # populate the following placeholders:
            # obs_t_ph
            # act_t_ph
            # rew_t_ph
            # obs_tp1_ph
            # done_mask_ph
            # (this is needed for computing total_error)
            # learning_rate -- you can get this from optimizer_spec.lr_schedule.value(t)
            # (this is needed by the optimizer to choose the learning rate)

            optimizer_lr = optimizer_spec.lr_schedule.value(t)

            feed_dict = {obs_t_ph: obs_t_batch,
                         act_t_ph: act_batch,
                         rew_t_ph: rew_batch,
                         obs_tp1_ph: obs_tp_batch,
                         done_mask_ph: done_batch,
                         learning_rate: optimizer_lr}
            error, _ = session.run([total_error, train_fn], feed_dict)

            # 3.d: periodically update the target network by calling
            # session.run(update_target_fn)
            # you should update every target_update_freq steps, and you may find the
            # variable num_param_updates useful for this (it was initialized to 0)
            if (t % target_update_freq == 0):
                print ("update of target network in progress ...")
                session.run(update_target_fn)
                num_param_updates += 1

            # YOUR CODE HERE

            #####

        ### 4. Log progress and store data for plotting
        x_list.append(t)
        avg_reward = np.mean(np.array(env.rew_mem))
        avg_score = np.mean(np.array(env.score_mem))
        avg_gain = np.mean(np.array(env.specs_mem['gain']))
        avg_bw = np.mean(np.array(env.specs_mem['bw']))
        avg_Ibias = np.mean(np.array(env.specs_mem['Ibias']))

        reward_list.append(avg_reward)
        score_list.append(avg_score)
        gain_list.append(avg_gain)
        bw_list.append(avg_bw)
        ibias_list.append(avg_Ibias)
        last_state_list.append(env.states_mem[-1])
        #print (t)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            print("Timestep %d" % (t,))
            print("last state tried {}" .format(env.states_mem[-1]))
            print("avg_reward %f" % avg_reward)
            print("avg_score %f" % avg_score)
            print("avg_gain %f" % avg_gain)
            print("avg_bw %f" % avg_bw)
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            print("error of Q_func estimator %f" % error)

            sys.stdout.flush()

        t += 1

    filename = "cs_amp.pickle"
    os.makedirs("./data", exist_ok=True)
    file_dir = os.path.join("data", filename)

    data = dict(
        t = x_list,
        last_state = last_state_list,
        reward = reward_list,
        score = score_list,
        gain = gain_list,
        bw = bw_list,
    )

    with open(file_dir, 'wb') as handle:
        pickle.dump(data, handle)

    plt.subplot(511)
    plt.plot(x_list, reward_list, lw=2)
    plt.xlabel("time steps")
    plt.ylabel("reward")
    plt.subplot(512)
    plt.plot(x_list, score_list, lw=2)
    plt.xlabel("time steps")
    plt.ylabel("score")
    plt.subplot(513)
    plt.plot(x_list, gain_list, lw=2)
    plt.xlabel("time steps")
    plt.ylabel("gain")
    plt.subplot(514)
    plt.plot(x_list, bw_list, lw=2)
    plt.xlabel("time steps")
    plt.ylabel("bw")
    plt.subplot(515)
    plt.plot(x_list, last_state_list, lw=2)
    plt.xlabel("time steps")
    plt.ylabel("last_state visited")
    plt.show()
