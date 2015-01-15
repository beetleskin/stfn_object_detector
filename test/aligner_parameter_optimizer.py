#!/usr/bin/env python

"""
    Copyright (c) 2013, Los Alamos National Security, LLC
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
    following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following
      disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
      following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Los Alamos National Security, LLC nor the names of its contributors may be used to endorse or
      promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
    INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
    WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
    THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from pyharmonysearch import ObjectiveFunctionInterface, harmony_search
import random
from bisect import bisect_left
from multiprocessing import cpu_count
import subprocess
import datetime



class ObjectiveFunction(ObjectiveFunctionInterface):

    def __init__(self):
        # all variables vary in the range [-100, 100]
        self._lower_bounds = [2, 1, 0, 0.01, 0.01, 100, 0.001, 0.005, 0.005]
        self._upper_bounds = [50, 200, 5, 10, 0.5, 500000, 0.05, 0.1, 0.1]
        self._variable = [True, True, True, True, True, True, True, True, True]

        # define all input parameters
        self._maximize = False  # minimize
        self._max_imp = 50000  # maximum number of improvisations
        self._hms = 250  # harmony memory size
        self._hmcr = 0.75  # harmony memory considering rate
        self._par = 0.5  # pitch adjusting rate
        self._mpap = 0.5  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only

    def get_fitness(self, vector):
        output = subprocess.check_output(["rosrun", "stfn_object_detector", "ros_aligner_fitness_checker", 
            "_a_numberOfSamples:=%.5f" % vector[0],
            "_a_correspondenceRandomness:=%.5f" % vector[1],
            "_a_similarityThreshold:=%.5f" % vector[2],
            "_a_maxCorrespondenceDistanceMultiplier:=%.5f" % vector[3],
            "_a_inlierFraction:=%.5f" % vector[4],
            "_a_maximumIterations:=%.5f" % vector[5],
            "_vg_leafSize:=%.5f" % vector[6],
            "_nest_radius:=%.5f" % vector[7],
            "_fest_radius:=%.5f" % vector[8]])

        error = float(output.split(":")[-1])
        return error


    def get_value(self, i, j=None):
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

    def get_lower_bound(self, i):
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        return self._upper_bounds[i]

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        return False

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return False

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    obj_fun = ObjectiveFunction()
    num_processes = cpu_count() - 1  # use number of logical CPUs - 1 so that I have one available for use
    #num_processes = 4
    num_iterations = num_processes  # each process does 1 iterations
    results = harmony_search(obj_fun, num_processes, num_iterations)
    result_output = 'Elapsed time: %s\nBest harmony: %s\nBest fitness: %s' % (results.elapsed_time, results.best_harmony, results.best_fitness)
    with open('hs_result_%s.txt' % start_time.strftime('%Y_%m_%d_%H_%M_%S'), 'w') as f: 
        f.write(result_output)
    print result_output
