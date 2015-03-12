#pragma once

#include <opencv2/core/core.hpp>

#include <vector>
#include <string>



struct Candidate {

	Candidate() : class_id(-1), confidence(-1) {
		bb.resize(2);
	}

	/* internal class id as stored in the forest*/
	int class_id;
	/* external class id (human readable name) */
	std::string class_name;
	/* the candidate's confidence value; the higher the better */
	float confidence;
	/* the candidates center */
	cv::Point center;
	/* the candidates bounding box */
	std::vector<cv::Point2f> bb;

	/* used internally for backprojection masking, is empty before and after processing, @see CRForestDetector::backprojectCandidate */
	cv::Mat backprojection_mask;
};
