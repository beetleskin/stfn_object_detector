#!/usr/bin/env python

from pybrain.optimization import GA
from aligner_evaluator import AlignerEvaluater
import time



if __name__ == '__main__':
	aEval = AlignerEvaluater(-1)
	bounds = zip(aEval.lower_bounds, aEval.upper_bounds)
	iterations = 10000
	start_time = time.time()
	optimizer = GA(aEval.evaluate, aEval.x0, verbose=True)

	try:
		optimizer.learn(iterations)
	except KeyboardInterrupt, e:
		pass

	elapsed_time = time.time() - start_time	
	best_params, best_error  = optimizer._bestFound()
	aEval.write_results("ga", list(best_params), best_error, iterations, elapsed_time)
