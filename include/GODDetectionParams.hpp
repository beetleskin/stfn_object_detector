#pragma once

#include <ros/ros.h>
#include <opencv2/core/core.hpp>

#include <memory>


#define PATH_SEP "/"


/**
 * Helper singleton class, wrapping most of the parameters needed for the Hough Random Forest detection and training.
 **/
class GODDetectionParams {

public:
	typedef std::shared_ptr<GODDetectionParams> Ptr;
	typedef std::shared_ptr<GODDetectionParams const> ConstPtr;

	static GODDetectionParams::Ptr get() {
		static GODDetectionParams::Ptr instance(new GODDetectionParams);
		return instance;
	}

protected:
	GODDetectionParams() {


		//
		// get the forest and tree paths
		this->database_path = "/home/stfn/recognition_database";
		this->forest_id = "foo";
		std::string tree_file_prefix = "treetable_";

		ros::param::get("~database_path", this->database_path);
		ros::param::get("~forest_id", this->forest_id);
		ros::param::get("~tree_file_prefix", tree_file_prefix);

		this->forest_path = database_path + PATH_SEP + std::string("forests") + PATH_SEP + forest_id;
		this->tree_path = forest_path + PATH_SEP + tree_file_prefix;


		//
		// get the models list
		ros::param::get("~models", this->models);


		//
		// get the number of trees
		this->number_of_trees = 15;
		ros::param::get("~number_of_trees", this->number_of_trees);


		//
		// get the patch size
		int patch_width = 10;
		int patch_height = 10;
		ros::param::get("~patch_width", patch_width);
		ros::param::get("~patch_height", patch_height);

		this->patch_size = cv::Size(patch_width, patch_height);


		//
		// get the image scale
		ros::param::get("~image_scale", this->image_scale);


		//
		// get the kernel sizes
		this->kernel_smooth = 20;
		this->kernel_vote = 20;
		this->kernel_sigma = 20;
		ros::param::get("~kernel_smooth", this->kernel_smooth);
		ros::param::get("~kernel_vote", this->kernel_vote);
		ros::param::get("~kernel_sigma", this->kernel_sigma);

		
		//
		// get the hierarchy threshold
		this->threshold_hierarchy = 1.5;
		ros::param::get("~threshold_hierarchy", this->threshold_hierarchy);


		//
		// get max candidate bounding box
		int cand_max_width = 320;
		int cand_max_height = 240;
		ros::param::get("~cand_max_width", cand_max_width);
		ros::param::get("~cand_max_height", cand_max_height);
		cand_max_width *= image_scale;
		cand_max_height *= image_scale;

		this->candidate_max_bb = cv::Size(cand_max_width, cand_max_height);


		//
		// get maximum number of detections per class
		this->max_candidates = 20;
		ros::param::get("~max_candidates", this->max_candidates);


		//
		// get backprojection flag
		this->do_backprojection = true;
		ros::param::get("~do_backprojection", this->do_backprojection);


		//
		// get the detection threshold
		this->detection_threshold = 0.1;
		ros::param::get("~detection_threshold", this->detection_threshold);


		//
		// get the random seed
		this->random_seed = 0;
		ros::param::get("~random_seed", this->random_seed);
		if(this->random_seed == 0) {
			this->random_seed = int(time(NULL));
		}



		nlabels = -1;
	}



public:
	/* the path to the database directory */
	std::string database_path;
	/* the forest ID */
	std::string forest_id;
	/* the path to the forest base directory */
	std::string forest_path;
	/* the tree base path, usually 'forest_path/treetable_' */
	std::string tree_path;
	/* the number of trees in the forest*/
	int number_of_trees;
	/* the external model IDs */
	std::vector<std::string> models;
	/* the width and height of a patch */
	cv::Size patch_size;
	/* input image scale: all input images are scaled with this factor, output data is scaled back, usually within [0.5, 1.0] */
	float image_scale;
	/* the maximum bounding box of a detected candidate */
	cv::Size candidate_max_bb;
	/* the kernel sizes used for image smoothing */
	float kernel_smooth;
	float kernel_vote;
	float kernel_sigma;
	/* the treshold used for hierarchy similarity */
	float threshold_hierarchy;
	/* number of maximum detections per class */
	float max_candidates;
	/* flag inidcating if backprojection of candidates is executed */
	bool do_backprojection;
	/* the theshold a detection needs to overcome */
	float detection_threshold;
	/* a random seed, yay! */
	int random_seed;


	// TODO: this is a shitty parameter
	float nlabels;
};