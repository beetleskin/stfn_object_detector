#pragma once

#include <opencv2/core/core.hpp>


struct GODImageMemory {

	/* The images store the leaf IDs for each pixel; One image per tree in the forest */
	static std::vector<cv::Mat> assign_images;

	/*  */
	static std::vector<cv::Mat> class_prior;

	/* The Hough space (per picel class posterior); One image per class (background excluded) */
	static std::vector<cv::Mat> hough_space;
};