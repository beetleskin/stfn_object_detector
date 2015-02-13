import sys
import json
import os
import subprocess

if __name__ == '__main__':

	results_dir = "test/results"
	for result_file in os.listdir(results_dir):
		with open(os.path.join(results_dir, result_file)) as f:


			print result_file


			result_data = json.load(f)
			params = result_data['parameter']
			cmd = ["rosrun", "stfn_object_detector", "ros_aligner_fitness_checker", 
				"_aligner_numberOfSamples:=%.5f" % params[0],
				"_aligner_correspondenceRandomness:=%.5f" % params[1],
				"_aligner_similarityThreshold:=%.5f" % params[2],
				"_aligner_maxCorrespondenceDistanceMultiplier:=%.5f" % params[3],
				"_aligner_inlierFraction:=%.5f" % params[4],
				"_aligner_maximumIterations:=%.5f" % params[5],
				"_aligner_vg_leafSize:=%.5f" % params[6],
				"_aligner_nest_radius:=%.5f" % params[7],
				"_aligner_fest_radius:=%.5f" % params[8]
			]

			summed_error = 0.0
			runs = 3
			for i in range(runs):
				process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
				output, _ = process.communicate()
				error = float(output.split(":")[-1])
				summed_error += error

			mean_error = summed_error / runs


			print "expected error: %f" % result_data['error']
			print "real error: %f\n" % mean_error






	print sys.argv