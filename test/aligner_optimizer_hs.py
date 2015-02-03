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
import sys

from aligner_evaluator import AlignerEvaluater



class ObjectiveFunction(ObjectiveFunctionInterface):

    def __init__(self, iterations=3000):
        self._aEval = AlignerEvaluater();
        self._lower_bounds = self._aEval.lower_bounds
        self._upper_bounds = self._aEval.upper_bounds
        self._variable = [True, True, True, True, True, True, True, True, True]

        # define all input parameters
        self._maximize = False  # minimize
        self._max_imp = iterations  # maximum number of improvisations
        self._hms = 200  # harmony memory size
        self._hmcr = 0.75  # harmony memory considering rate
        self._par = 0.5  # pitch adjusting rate
        self._mpap = 0.5  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only


    def get_fitness(self, vector):
        return self._aEval.evaluate(vector)


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
    iterations = 1000
    if len(sys.argv) > 1:
        iterations = int(sys.argv[1])
    obj_fun = ObjectiveFunction(iterations)
    #num_processes = cpu_count() - 1  # use number of logical CPUs - 1 so that I have one available for use
    num_processes = 4
    num_iterations = num_processes  # each process does 1 iterations
    results = harmony_search(obj_fun, num_processes, num_iterations)

    obj_fun._aEval.write_results("hs", results.best_harmony, results.best_fitness, -1, results.elapsed_time.seconds)
