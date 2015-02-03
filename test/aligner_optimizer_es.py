#!/usr/bin/env python
from aligner_evaluator import AlignerEvaluater

from pybrain.optimization import ES
import sys
import time




if __name__ == '__main__':
	iterations = 1000
	if len(sys.argv) > 1:
		iterations = int(sys.argv[1])
	aEval = AlignerEvaluater(-1)
	bounds = zip(aEval.lower_bounds, aEval.upper_bounds)
	start_time = time.time()
	optimizer = ES(aEval.evaluate, aEval.x0, evaluatorIsNoisy=True, verbose=True)

	try:
		optimizer.learn(iterations)
	except KeyboardInterrupt, e:
		pass

	elapsed_time = time.time() - start_time	
	best_params, best_error  = optimizer._bestFound()
	aEval.write_results("es", list(best_params), best_error, iterations, elapsed_time)
