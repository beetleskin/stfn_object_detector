#include "CRPatch.hpp"
#include "GpuHoG.hpp"
#include "myutils.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


HoG CRPatch::hog;


void CRPatch::extractPatches(Mat &img, Mat &depth_img, unsigned int n, int label, int imageID, Point &object_center) {
	// extract features
	vector<Mat> features;
	extractFeatureChannels(img, depth_img, features);


	// generate x,y locations
	CvMat *locations = cvCreateMat( (img.cols - 2*patch_size.width) * (img.rows - 2*patch_size.height), 1, CV_32SC2 );
	cvRandArr( cvRNG, locations, CV_RAND_UNI, cvScalar(0, 0, 0, 0), cvScalar(img.cols - 2*patch_size.width, img.rows - 2*patch_size.height, 0, 0) );

	// reserve memory
	unsigned int offset = vLPatches[label].size();
	vLPatches[label].reserve(offset + n);
	for (unsigned int i = 0; i < n ; ++i) {

		CvPoint pt = *(CvPoint *)cvPtr1D( locations, i, 0 );

		PatchFeature pf;
		vLPatches[label].push_back(pf);
		vImageIDs[label].push_back(imageID);// adding the image id to the patch

		vLPatches[label].back().roi.x = pt.x;
		vLPatches[label].back().roi.y = pt.y;
		vLPatches[label].back().roi.width = 2*patch_size.width;
		vLPatches[label].back().roi.height = 2*patch_size.height;

		vLPatches[label].back().center.x = pt.x + patch_size.width - object_center.x;
		vLPatches[label].back().center.y = pt.y + patch_size.height - object_center.y;

		vLPatches[label].back().vPatch.resize(features.size());
		for (unsigned int c = 0; c < features.size(); ++c) {
			vLPatches[label].back().vPatch[c] = features[c](vLPatches[label].back().roi).clone();
		}
	}
}



void CRPatch::extractFeatureChannels(Mat &img, Mat &depth_img, std::vector<Mat> &features) {
	Mat I_x, I_y;
	features.resize(61);
	for (unsigned int c = 0; c < features.size(); ++c)
		features[c] = Mat::zeros(img.size(), CV_8UC1);


	// Get intensity
	cvtColor( img, features[0], CV_RGB2GRAY );

	// |I_x|, |I_y|
	Sobel(features[0], I_x, CV_16SC1, 1, 0, 3);
	Sobel(features[0], I_y, CV_16SC1, 0, 1, 3);
	convertScaleAbs( I_x, features[3], 0.25);
	convertScaleAbs( I_y, features[4], 0.25);

	// 9-bin HOG feature stored at features[7] - features[15]
	GpuHoG gpuHog;
	vector<Mat> featuresHog(features.begin() + 7, features.begin() + 7 + 9);
	gpuHog.compute(features[0], featuresHog);

	// |I_xx|, |I_yy|
	Sobel(features[0], I_x, CV_16SC1, 2, 0, 3);
	Sobel(features[0], I_y, CV_16SC1, 0, 2, 3);
	convertScaleAbs( I_x, features[5], 0.25);
	convertScaleAbs( I_y, features[6], 0.25);

	// L, a, b
	cvtColor(img, img, CV_RGB2Lab);
	split(img, vector<Mat>(features.begin(), features.begin() + 3));

	// min filter
	for (int c = 0; c < 16; ++c) {
		erode(features[c], features[c + 16], Mat(5, 5, CV_8UC1));
		dilate(features[c], features[c], Mat(5, 5, CV_8UC1));
	}



	// Depth HoG
	Sobel(depth_img, I_x, CV_32FC1, 1, 0, 3);
	Sobel(depth_img, I_y, CV_32FC1, 0, 1, 3);
	I_x /= 8.f;
	I_y /= 8.f;

	// depth gradient orientation and magnitude
	for (int y = 0; y < depth_img.rows; ++y) {
		float *grad_x = I_x.ptr<float>(y);
		float *grad_y = I_y.ptr<float>(y);
		uchar *grad_orient = features[32].ptr<uchar>(y);
		uchar *grad_mag = features[33].ptr<uchar>(y);

		for (int x = 0; x < depth_img.cols; ++x) {
			// Orientation of gradients
			float tx = grad_x[x] + _copysign(0.000001f, grad_x[x]);
			// Scaling [-pi pi] -> [0 80*pi]
			grad_orient[x] = uchar( (atan2(grad_y[x], grad_x[x]) + M_PI) * 40 );

			// Magnitude of gradients
			float mag = sqrt(grad_x[x]*grad_x[x] + grad_y[x]*grad_y[x]) * 5;
			grad_mag[x] = uchar( (mag > 255) ? 255 : mag );
		}
	}

	// 9-bin HOG feature stored at features[7] - features[15]
	vector<Mat> featuresDepthHog(features.begin() + 37, features.begin() + 37 + 9);
	hog.extractOBin(features[32], features[33], featuresDepthHog);


	// |dI_x|, |I_y|
	I_x = abs(I_x);
	I_x *= 5;
	threshold(I_x, I_x, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(I_x, features[33], 1);
	I_y = abs(I_y);
	I_y *= 5;
	threshold(I_y, I_y, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(I_y, features[34], 1);

	// |I_xx|, |I_yy|
	Sobel(depth_img, I_x, CV_32FC1, 2, 0, 5);
	Sobel(depth_img, I_y, CV_32FC1, 0, 2, 5);
	I_x /= 64.f;
	I_y /= 64.f;

	I_x = abs(I_x);
	I_x *= 5;
	threshold(I_x, I_x, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(I_x, features[35], 1);
	I_y = abs(I_y);
	I_y *= 5;
	threshold(I_y, I_y, 255, 255, CV_THRESH_TRUNC);
	convertScaleAbs(I_y, features[36], 1);

	// true depth
	depth_img.convertTo(depth_img, CV_32FC1);
	Mat points(depth_img.size(), CV_32FC3);
	Mat dist(depth_img.size(), CV_32FC1);
	calcPoints(depth_img, points);
	for (int y = 0; y < points.rows; ++y) {
		Eigen::Vector3f *r_ptr_points = points.ptr<Eigen::Vector3f>(y);
		float *r_ptr_dist = dist.ptr<float>(y);
		for (int x = 0; x < points.cols; ++x) {
			r_ptr_dist[x] = r_ptr_points[x].norm();
		}
	}
	features[60] = dist;


	// scaled depth value
	float max = 5000.f;
	for (size_t y = 0; y < depth_img.rows; ++y) {
		float *row_ptr = depth_img.ptr<float>(y);
		for (size_t x = 0; x < depth_img.cols; ++x) {
			if (row_ptr[x] > max)
				row_ptr[x] = max;
			else if (row_ptr[x] <= 0)
				row_ptr[x] == std::numeric_limits<uchar>::quiet_NaN();

		}
	}
	cv::convertScaleAbs(depth_img, features[32], 255.0/max);


	// min filter
	for (int c = 32; c < 46; ++c) {
		erode(features[c], features[c + 14], Mat(5, 5, CV_8UC1));
		dilate(features[c], features[c], Mat(5, 5, CV_8UC1));
	}
}
