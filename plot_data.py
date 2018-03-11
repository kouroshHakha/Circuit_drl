import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


if __name__ == '__main__':

    data_dir = "./data/cs_amp.pickle"
    saved_data = None
    with open(data_dir, 'rb') as f:
        saved_data = pickle.load(f)
    if saved_data is not None:
        indexes = np.arange(1,1000,1)
        t = np.array(saved_data['t'])[indexes]
        last_state= np.array(saved_data['last_state'])[indexes]
        reward = np.array(saved_data['reward'])[indexes]
        score = np.array(saved_data['score'])[indexes]
        gain = np.array(saved_data['gain'])[indexes]
        bw = np.array(saved_data['bw'])[indexes]
        print(gain)
        print(bw)
        print(score)

        plt.subplot(511)
        plt.plot(t, reward, lw=2)
        plt.xlabel("time steps")
        plt.ylabel("reward")
        plt.subplot(512)
        plt.plot(t, score, lw=2)
        plt.xlabel("time steps")
        plt.ylabel("score")
        plt.subplot(513)
        plt.plot(t, gain, lw=2)
        plt.xlabel("time steps")
        plt.ylabel("gain")
        plt.subplot(514)
        plt.plot(t, bw, lw=2)
        plt.xlabel("time steps")
        plt.ylabel("bw")
        plt.subplot(515)
        plt.plot(t, last_state, lw=2)
        plt.xlabel("time steps")
        plt.ylabel("last_state visited")
        plt.show()

