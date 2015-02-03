#!/usr/bin/env python
from aligner_evaluator import AlignerEvaluater

from pybrain.optimization import GA
import sys
import time




if __name__ == '__main__':
	iterations = 1000
	if len(sys.argv) > 1:
		iterations = int(sys.argv[1])
	aEval = AlignerEvaluater(-1)
	start_time = time.time()
	optimizer = GA(aEval.evaluate, aEval.x0, verbose=True)

	try:
		optimizer.learn(iterations)
	except KeyboardInterrupt, e:
		pass

	elapsed_time = time.time() - start_time	
	best_params, best_error  = optimizer._bestFound()
	aEval.write_results("ga", list(best_params), best_error, iterations, elapsed_time)
