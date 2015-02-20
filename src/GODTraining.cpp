#include "GODTraining.hpp"

#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>

#define PATH_SEP "/"

GODTraining::GODTraining() {
	// set number of cores for multiprocessing
	tbb::task_scheduler_init init(5);

	// initialize the detector with ros parameters
	initTraining();
}

GODTraining::~GODTraining() {
}


void GODTraining::initTraining() {
	// default values for params
	nlabels = -1;

	// temp params
	std::string database_path = "/home/stfn/recognition_database";
	std::string forest_id = "foo";
	int num_trees = 15;
	int off_trees = 0;

	// override params from ros parameter server
	ros::param::get("~database_path", database_path);
	ros::param::get("~forest_id", forest_id);

	ros::param::get("~num_trees", num_trees);
	ros::param::get("~initial_scale", initial_scale);

	

	// init forest
	crForest.reset(new CRForest(num_trees));
	crForest->training_mode = 1;

	// Create directory
	string forest_path = database_path + PATH_SEP + string("forests") + PATH_SEP + forest_id + PATH_SEP;
	string execstr = "mkdir ";
	execstr += forest_path;
	cout << execstr << endl;
	int ret = system( execstr.c_str() );
}


// Extract patches from training data
void GODTraining::load_traindata(CRPatch &Train, CvRNG *pRNG) {

/*
	map<int, string> model_map;
	vector<string> background_map;
	ros::param::get("~models", model_map);
	ros::param::get("~backgrounds", background_map);

	cout << model_map[0] << endl;
	cout << background_map[0] << endl;


	vector<vector<string> > vFilenames;
	vector<vector<Rect> > vBBox;
	vector<vector<Point> > vCenter;

	*/

/*
	// load positive file list
	loadTrainClassFile(trainclassfiles, vFilenames,  vBBox, vCenter);
	Train.setClasses(nlabels);


	// for each class/label
	for (int l = 0; l < nlabels; ++l) {

		cout << "Label: " << l << " " << class_structure[l] << " ";

		int subsamples = 0;
		if (class_structure[l] == 0)
			subsamples = subsamples_class_neg;
		else
			subsamples = subsamples_class;

		// load postive images and extract patches
		for (int i = 0; i < (int)vFilenames[l].size(); ++i) {

			if (i % 50 == 0) cout << i << " " << flush;

			if (subsamples <= 0 || (int)vFilenames[l].size() <= subsamples || (cvRandReal(pRNG)*double(vFilenames[l].size()) < double(subsamples)) ) {

				// Load image
				Mat img, depth_img;

				string img_file_name = vFilenames[l][i];
				string img_depth_file_name = img_file_name;
				int start_pos = img_file_name.find("crop.png");
				if (start_pos != -1) {
					img_depth_file_name.replace(start_pos, 8, "depthcrop.png");
				} else {
					start_pos = img_file_name.find(".png");
					img_depth_file_name.replace(start_pos, 4, "_depth.png");
				}


				img = imread(trainclasspath + PATH_SEP + img_file_name, CV_LOAD_IMAGE_COLOR);
				// is going to be IPL_DEPTH_16U
				depth_img = imread(trainclasspath + PATH_SEP + img_depth_file_name, CV_LOAD_IMAGE_ANYDEPTH);

				if (!img.data) {
					cout << "Could not load image file: " << trainclasspath + PATH_SEP + img_file_name << endl;
					exit(-1);
				} else if (!depth_img.data) {
					cout << "Could not load image file: " << trainclasspath + PATH_SEP + img_depth_file_name << endl;
					exit(-1);
				}

				// resize
				float r_scale = 0.7;
				resize(img, img, Size(), r_scale, r_scale, CV_INTER_LINEAR);
				resize(depth_img, depth_img, Size(), r_scale, r_scale, CV_INTER_NN);
				vBBox[l][i].x *= r_scale;
				vBBox[l][i].y *= r_scale;
				vBBox[l][i].width *= r_scale;
				vBBox[l][i].height *= r_scale;
				vCenter[l][i].x *= r_scale;
				vCenter[l][i].y *= r_scale;

				// Extract positive training patches
				Train.extractPatches(img, depth_img, samples_class, l, i , vBBox[l][i], vCenter[l][i]);
			}
		}
	}*/
}



void GODTraining::train() {
	int p_width = 10;
	int p_height = 10;

	ros::param::get("~p_width", p_width);
	ros::param::get("~p_height", p_height);

	// create random seed
	time_t t = time(NULL);
	int seed = (int)t;
	ros::param::get("~random_seed", seed);
	CvRNG cvRNG(seed);


	// Init training data
	CRPatch Train(&cvRNG, p_width, p_height);

	// Extract training patches
	load_traindata(Train, &cvRNG);
}