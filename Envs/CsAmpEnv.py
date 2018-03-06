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

class CsAmpEnv(object):

    def __init__(self, num_process, design_netlist, target_specs=None):

        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.num_process = num_process
        self.base_tmp_dir = os.path.abspath("/tmp/circuit_drl")
        self.gen_dir = os.path.join(self.base_tmp_dir, "designs_" + self.base_design_name)

        os.makedirs(self.base_tmp_dir, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        raw_file = open(design_netlist, 'r')
        self.tmp_lines = raw_file.readlines()
        raw_file.close()

        self.target_specs = target_specs

    def get_design_name(self, state):
        fname = self.base_design_name
        for keyword, value in state.items():
            fname += "_" + keyword + "_" + str(value)
        return fname

    def create_design(self, state):
        new_fname = self.get_design_name(state)
        design_folder = os.path.join(self.gen_dir, new_fname)
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.cir')

        # if os.path.exists(fpath):
        #     print ("design already exists, no need to generate one. skipping create_design() ...")
        #     return fpath

        lines = copy.deepcopy(self.tmp_lines)
        for line_num, line in enumerate(lines):
            if '.param' in line:
                for key, value in state.items():
                    regex = re.compile("%s=(\S+)" % (key))
                    found = regex.search(line)
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)
            if 'wrdata' in line:
                regex = re.compile("wrdata\s*(\w+\.\w+)\s*")
                found = regex.search(line)
                if found :
                    replacement = os.path.join(design_folder, found.group(1))
                    lines[line_num] = lines[line_num].replace(found.group(1), replacement)

        with open(fpath, 'w') as f:
            f.writelines(lines)
            f.close()
        return design_folder, fpath

    def simulate(self, fpath):
        command = "ngspice -b %s >/dev/null 2>&1" %fpath
        #print("command",command)
        #os.system("cat %s" %fpath)
        os.system(command)

    def create_design_and_simulate(self, state):
        dsn_name = self.get_design_name(state)
        design_folder, fpath = self.create_design(state)
        self.simulate(fpath)
        result = self.get_score(design_folder)
        return state, result


    def run(self, states):
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state) for state in states]
        results = pool.map(self.create_design_and_simulate, arg_list)
        return results

    def get_score(self, output_path):
        """

        :param output_path:
        :return
        score:  a single scalar value representing the value of the current state
                reward will essentially be the change in score
        terminate: true if we met all the specs
        """

        bw_min = self.target_specs['bw_min']
        gain_min = self.target_specs['gain_min']
        terminate = False
        # use parse output here and also the self.target_specs dictionary that user has provided
        freq, vout, Ibias = self.parse_output(output_path)
        bw = self.find_bw(vout, freq)
        gain = self.find_dc_gain(vout)

        if (bw > bw_min and gain > gain_min):
            score = 1/Ibias
            terminate = True
        else:
            score = - (abs(bw - bw_min) / bw_min + abs(gain - gain_min) / gain_min)
            
        # print('bw', bw)
        # print('gain', gain)
        # print ('Ibias', Ibias)

        spec = dict(
            bw=bw,
            gain=gain,
            Ibias=Ibias
        )
        return score, terminate, spec

    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, 'ac.csv')
        dc_fname = os.path.join(output_path, 'dc.csv')

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout = ac_raw_outputs[:, 1]
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias



    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_bw(self, vout, freq):
        gain = np.abs(vout)
        gain_3dB = gain[0] / np.sqrt(2)
        return self._get_best_crossing(freq, gain, gain_3dB)


    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop

def generate_random_state (len):
    states = []
    for _ in range(len):
        vbias = random.random() * 1.8
        mul = int(random.random() * (100 - 1) + 1)
        rload = random.random() * (1000 - 10) + 10
        #cload = random.random() * (1e-12 - 1e-15) + 1e-15
        states.append(dict(
            vbias=vbias,
            mul=mul,
            rload=rload,
            #cload=cload
        ))
    return states

if __name__ == '__main__':

    num_process = 1
    num_designs = 1
    dsn_netlist = '../netlists/cs_amp.cir'
    target_spec = dict(gain_min=3.5, bw_min=1e9)

    cs_env = CsAmpEnv(num_process=num_process,
                      design_netlist=dsn_netlist,
                      target_specs=target_spec)
    states = generate_random_state(num_designs)
    #states = [{'vbias': 0.7,
    #           'mul': 12,
    #           'rload': 400,
    #           'cload': 50e-13}]

    start_time = time.time()
    results = cs_env.run(states)
    end_time = time.time()

    #pprint.pprint(results)
    print("time for num_process=%d, num_designs=%d : %f" % (num_process, num_designs, end_time - start_time))
    
    best_score = -np.float('inf')
    for result in results:
        score = result[1][0]
        if score > best_score:
            best_score = score
            best_state = result[0]
            best_spec = result[1][2]
    #for i,value in results.enumerate():
    #    print(value)
    print (type(results))    
    print(results)
    print(best_score)
    print(best_state)
    print(best_spec)
