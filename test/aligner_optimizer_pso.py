#!/usr/bin/env python
from aligner_evaluator import AlignerEvaluater

from pybrain.optimization import ParticleSwarmOptimizer
import sys
import time



if __name__ == '__main__':
	iterations = 1000
	if len(sys.argv) > 1:
		iterations = int(sys.argv[1])
	aEval = AlignerEvaluater()
	bounds = zip(aEval.lower_bounds, aEval.upper_bounds)
	start_time = time.time()
	optimizer = ParticleSwarmOptimizer(aEval.evaluate, aEval.x0, boundaries=bounds, minimize=True, verbose=True)

	try:
		optimizer.learn(iterations)
	except KeyboardInterrupt, e:
		pass

	elapsed_time = time.time() - start_time	
	best_params, best_error  = optimizer._bestFound()
	aEval.write_results("pso", list(best_params), best_error, iterations, elapsed_time)
