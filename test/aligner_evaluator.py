import datetime
import json
import subprocess
import threading



class AlignerEvaluater(object):
	
	def __init__(self, error_sign=1):
		self._error_sign = error_sign
		self.start_time = datetime.datetime.now()
		self._lower_bounds = [2, 1, 0.001, 0.01, 0.01, 100, 0.001, 0.005, 0.005]
		self._upper_bounds = [50, 200, 1, 10, 0.5, 500000, 0.05, 0.1, 0.1]
		self._x0 = [3, 50, 0.5, 0.3, 0.0, 50000, 0.01, 0.01, 0.025]
		self._num_evals = 0


	def evaluate(self, params):
		cmd = ["rosrun", "stfn_object_detector", "ros_aligner_fitness_checker", 
				"_a_numberOfSamples:=%.5f" % params[0],
				"_a_correspondenceRandomness:=%.5f" % params[1],
				"_a_similarityThreshold:=%.5f" % params[2],
				"_a_maxCorrespondenceDistanceMultiplier:=%.5f" % params[3],
				"_a_inlierFraction:=%.5f" % params[4],
				"_a_maximumIterations:=%.5f" % params[5],
				"_vg_leafSize:=%.5f" % params[6],
				"_nest_radius:=%.5f" % params[7],
				"_fest_radius:=%.5f" % params[8]]

		def target():
			self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
			self.current_output, _ = self.process.communicate()

		try:
			thread = threading.Thread(target=target)
			thread.start()

			thread.join(10)
			if thread.is_alive():
				self.process.terminate()
				thread.join()

			if(self.current_output):
				error = float(self.current_output.split(":")[-1])
			else:
				error = 100
		except subprocess.CalledProcessError:
			error = 100

		self._num_evals += 1
		print "[%d] %.4f" % (self._num_evals, error)
		return self._error_sign*error


	def write_results(self, optimizer_name, best_params, best_error, iterations=-1, elapsed=-1):
		result = {
			"parameter": best_params,
			"error": best_error,
			"iterations": iterations,
			"elapsed": elapsed
		}
		result_str = json.dumps(result, indent=4, separators=(',', ': '))

		print "\n" + result_str
		with open('results/opti_result_%s_%s.txt' % (optimizer_name, self.start_time.strftime('%Y_%m_%d_%H_%M')), 'w') as f: 
			f.write(result_str)


	@property
	def lower_bounds(self):
		return self._lower_bounds

	@property
	def upper_bounds(self):
		return self._upper_bounds

	@property
	def x0(self):
		return self._x0
