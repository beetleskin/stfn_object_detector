#include "GODTraining.hpp"

#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>

#define PATH_SEP "/"



GODTraining::GODTraining() {
	// set number of cores for multiprocessing
	tbb::task_scheduler_init init(5);

	// initialize the training with ros parameters
	this->params = GODDetectionParams::get();
	initTraining();
}

GODTraining::~GODTraining() {
}


void GODTraining::initTraining() {

	//
	// get parameters from world model
	int number_of_trees = params->number_of_trees;
	std::string forest_path = params->forest_path;


	// init forest
	forest_ptr.reset(new CRForest(number_of_trees));

	// Create directory
	std::string make_cmd = string("mkdir ") + forest_path;
	int system_status = system(make_cmd.c_str());
	
	// check if the directory was created. ask the user to delete it, if it exists
	if(system_status != 0) {

		bool exit = false;
		char choice;

		cout << "Forest exists, override? [y/n]: ";
		cin >> choice;

		if(choice == 'y') {
			string rm_cmd = string("rm -r ") + forest_path;
			system_status = system(rm_cmd.c_str());

			if(system_status != 0) { 
				exit = true;
			} else {
				system_status = system_status = system(make_cmd.c_str());
				if(system_status != 0){
					exit = true;
				}
			}
		} else {
			exit = true;
		}


		if(exit){
			ROS_ERROR("Could not create forest path, system exited with: %d", system_status);
			ros::shutdown();
		}
	}
}


// Extract patches from training data
std::map<std::string, int> GODTraining::load_traindata(CRPatch &Train, CvRNG *pRNG) {

	//
	// get parameters from world model
	std::string database_path = params->database_path;
	float image_scale = params->image_scale;
	vector<string> models = params->models;

	//
	// get additional params from ROS
	
	vector<string> backgrounds;
	int subsample_models = -1;
	int subsample_backgrounds = -1;
	int patches_per_model_image = 100;
	int patches_per_background_image = 250;
	
	ros::param::get("~backgrounds", backgrounds);
	ros::param::get("~subsample_models", subsample_models);
	ros::param::get("~subsample_backgrounds", subsample_backgrounds);
	ros::param::get("~patches_per_model_image", patches_per_model_image);
	ros::param::get("~patches_per_background_image", patches_per_background_image);

	std::map<std::string, vector<string> > image_files;
	std::map<std::string, vector<cv::Rect> > object_bbs;
	std::map<std::string, vector<cv::Point> > object_centers;
	std::map<std::string, int> class_map;



	//
	// parse the config files from the database for each model

	for (int i = 0; i < models.size(); ++i) {
		std::string model_key = models[i];
		string model_config_file = database_path + PATH_SEP + "models" + PATH_SEP + models[i] + PATH_SEP + models[i] + ".cfg";
		ifstream infile(model_config_file.c_str());
		
		if (infile.is_open()) {
			string image_file;
			cv::Rect bb;
			cv::Point center;
			while (infile >> image_file >> bb.x >> bb.y >> bb.width >> bb.height >> center.x >> center.y ) {
			    image_files[model_key].push_back(image_file);
			    object_bbs[model_key].push_back(bb);
			    object_centers[model_key].push_back(center);
			}
		} else {
			ROS_ERROR("Could not open config file: %s", model_config_file.c_str());
		}

		// assign internal class id
		class_map[model_key] = i+1;
	}

	// subsample models
	if(subsample_models) {
		for (int i = 0; i < models.size(); ++i) {
			std::string model_key = models[i];
			while(image_files[model_key].size() > subsample_models) {
				int idx_to_remove = (int) (cvRandReal(pRNG) * image_files[model_key].size());
				image_files[model_key].erase(image_files[model_key].begin() + idx_to_remove);
				object_bbs[model_key].erase(object_bbs[model_key].begin() + idx_to_remove);
				object_centers[model_key].erase(object_centers[model_key].begin() + idx_to_remove);
			}
		}
	}


	//
	// parse the config files from the database for each background

	for (int i = 0; i < backgrounds.size(); ++i) {
		std::string background_key = backgrounds[i];
		string scene_config_file = database_path + PATH_SEP + "scenes" + PATH_SEP + backgrounds[i] + PATH_SEP + backgrounds[i] + ".cfg";
		ifstream infile(scene_config_file.c_str());
		
		if (infile.is_open()) {
			string image_file;
			while (infile >> image_file) {
				image_files[background_key].push_back(image_file);
			}
		} else {
			ROS_ERROR("Could not open config file: %s", scene_config_file.c_str());
		}

		// assign internal class id
		class_map[background_key] = 0;
	}

	// subsample background
	if(subsample_models) {
		for (int i = 0; i < backgrounds.size(); ++i) {
			std::string background_key = backgrounds[i];
			while(image_files[background_key].size() > subsample_models) {
				int idx_to_remove = (int) (cvRandReal(pRNG) * image_files[background_key].size());
				image_files[background_key].erase(image_files[background_key].begin() + idx_to_remove);
				//object_bbs[background_key].erase(object_bbs[background_key].begin() + idx_to_remove);
				//object_centers[background_key].erase(object_centers[background_key].begin() + idx_to_remove);
			}
		}
	}


	// set number of labels; background has label 0, no matter how many background datasets there are
	int number_of_labels = models.size() + int(bool(backgrounds.size()));
	Train.setClasses(number_of_labels);


	//
	// extract patches from the images
	for ( const auto &pair : class_map ) {
		std::string class_key = pair.first;
		int class_id = pair.second;
		cout << pair.first << " " << pair.second << endl;


		for (int i = 0; i < image_files[class_key].size(); ++i) {

			
			std::string img_file_name = image_files[class_key][i];
			std::string img_depth_file_name = img_file_name;


			// get the correct image names for depth and rgb images
			int start_pos = img_file_name.find("crop.png");
			if (start_pos != -1) {
				img_depth_file_name.replace(start_pos, 8, "depthcrop.png");
			} else {
				start_pos = img_file_name.find(".png");
				img_depth_file_name.replace(start_pos, 4, "_depth.png");
			}


			// load images
			Mat img, depth_img;
			img = imread(img_file_name, CV_LOAD_IMAGE_COLOR);
			// is going to be IPL_DEPTH_16U
			depth_img = imread(img_depth_file_name, CV_LOAD_IMAGE_ANYDEPTH);

			if (!img.data) {
				cout << "Could not load image file: " << img_file_name << endl;
				exit(-1);
			} else if (!depth_img.data) {
				cout << "Could not load image file: " << img_depth_file_name << endl;
				exit(-1);
			}

			// downscale rgb and depth image
			resize(img, img, Size(), image_scale, image_scale, CV_INTER_LINEAR);
			resize(depth_img, depth_img, Size(), image_scale, image_scale, CV_INTER_NN);

			if(class_id != 0) {
				cv::Point &center = object_centers[class_key][i];
				center.x *= image_scale;
				center.y *= image_scale;

				Train.extractPatches(img, depth_img, patches_per_model_image, class_id, i, center);
			} else {
				cv::Point center;
				Train.extractPatches(img, depth_img, patches_per_background_image, class_id, i , center);
			}
			

			// Extract positive training patches
			
		}

    }


    return class_map;
}



void GODTraining::train() {
	//
	// get parameters from world model
	cv::Size patch_size = params->patch_size;
	std::string tree_path = params->tree_path;
	CvRNG cvRNG(params->random_seed);


	//
	// get additional params from ROS
	// create random seed
	int tree_max_depth = 20;
	ros::param::get("~tree_max_depth", tree_max_depth);


	// Init training data
	CRPatch Train(&cvRNG, patch_size);

	// Extract training patches
	std::map<std::string, int> class_map = load_traindata(Train, &cvRNG);


	// dispensable params ...
	std::vector<int> class_structure;
	bool has_background = false;
	for ( const auto &pair : class_map ) {
		int class_id = pair.second;
		if(class_id != 0)
			class_structure.push_back(pair.second);
		else
			has_background = true;
	}
	if(has_background)
		class_structure.push_back(0);


	// Train forest
	forest_ptr->trainForest(20, tree_max_depth, &cvRNG, Train, 2000, class_structure);


	// dispensable: initializing some statistics in to the leaf nodes
	for (unsigned int trNr = 0; trNr < forest_ptr->getTrees().size(); ++trNr) {
		LeafNode *leaf = forest_ptr->getTrees()[trNr]->getLeaf();
		LeafNode *ptLN = &leaf[0];
		vector<int> class_ids;
		forest_ptr->getTrees()[trNr]->getClassId(class_ids);

		for (unsigned int lNr = 0 ; lNr < forest_ptr->getTrees()[trNr]->getNumLeaf(); lNr++, ++ptLN) {
			ptLN->eL = 0;
			ptLN->fL = 0;
			ptLN->vLabelDistrib.resize(class_ids.size(), 0);
		}
	}


	// Save forest
	forest_ptr->saveForest(tree_path.c_str(), 0);
}