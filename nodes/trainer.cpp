#include "GODTraining.hpp"


GODTraining::Ptr trainer;


int main(int argc, char **argv) {
	ros::init(argc, argv, "detector");

	// initialize detector
	trainer.reset(new GODTraining);
	trainer->train();

	ros::spinOnce();
	return 0;
}