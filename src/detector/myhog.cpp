#include <string>
#include <iostream>
#include <boost/progress.hpp>
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "HoG.hpp"
#include "GpuHoG.hpp"

using namespace std;
using namespace cv;


string src_file = "/home/stfn/dev/rgbd-dataset/rgbd-scenes/background/background_10/background_10_39.png";

Mat get_hogdescriptor_visual_image(Mat &origImg, Mat &descriptorValues, Size winSize, Size cellSize, int scaleFactor, double viz_factor) {

	Mat visual_image;
	resize(origImg, visual_image, Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));

	int gradientBinSize = 9;
	// dividing 180° into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14 / (float)gradientBinSize;

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
	int cells_in_y_dir = winSize.height / cellSize.height;
	int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
	float *** gradientStrengths = new float **[cells_in_y_dir];
	int **cellUpdateCounter   = new int *[cells_in_y_dir];
	for (int y = 0; y < cells_in_y_dir; y++) {
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x < cells_in_x_dir; x++) {
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin < gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}
	Mat m = Mat::zeros(cells_in_y_dir, cells_in_x_dir, CV_8UC1);

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	float *descriptorDataIdx = descriptorValues.ptr<float>();
	int cellx = 0;
	int celly = 0;

	for (int blockx = 0; blockx < blocks_in_x_dir; blockx++) {
		for (int blocky = 0; blocky < blocks_in_y_dir; blocky++) {
			// 4 cells per block ...
			for (int cellNr = 0; cellNr < 4; cellNr++) {
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr == 1) celly++;
				if (cellNr == 2) cellx++;
				if (cellNr == 3) {
					cellx++;
					celly++;
				}

				m.at<uchar>(celly, cellx) += 1;

				for (int bin = 0; bin < gradientBinSize; bin++) {
					float gradientStrength = *descriptorDataIdx;
					descriptorDataIdx++;
					if (cellx == 1 && celly == 1)
						cout << (float)*descriptorDataIdx << "\t";

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)
				if (cellx == 1 && celly == 1)
					cout << endl;


				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)


		} // for (all block x pos)
	} // for (all block y pos)

	cout << m << endl;
	// compute average gradient strengths
	for (int celly = 0; celly < cells_in_y_dir; celly++) {
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

			float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin < gradientBinSize; bin++) {
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}


	cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

	// draw cells
	for (int celly = 0; celly < cells_in_y_dir; celly++) {
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;

			int mx = drawX + cellSize.width / 2;
			int my = drawY + cellSize.height / 2;

			rectangle(visual_image,
			          Point(drawX * scaleFactor, drawY * scaleFactor),
			          Point((drawX + cellSize.width)*scaleFactor,
			                (drawY + cellSize.height)*scaleFactor),
			          CV_RGB(100, 100, 100),
			          1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin < gradientBinSize; bin++) {
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos( currRad );
				float dirVecY = sin( currRad );
				float maxVecLen = cellSize.width / 2;
				float scale = viz_factor; // just a visual_imagealization scale,
				// to see the lines better

				// compute line coordinates
				float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				line(visual_image,
				     Point(x1 * scaleFactor, y1 * scaleFactor),
				     Point(x2 * scaleFactor, y2 * scaleFactor),
				     CV_RGB(0, 0, 255),
				     1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)


	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y < cells_in_y_dir; y++) {
		for (int x = 0; x < cells_in_x_dir; x++) {
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visual_image;

}

void hogIt(Mat &img) {
	// grayscale
	Mat img_gray, tmp;
	resize(img, tmp, Size(), 0.5, 0.5);
	img = tmp.clone();
	cvtColor(img, img_gray, CV_BGR2GRAY);

	{
		boost::progress_timer t;
		for (int i = 0; i < 100; ++i) {
			vector<Mat> vImgHog;
			for (int i = 0; i < 9; ++i) {
				vImgHog.push_back(Mat::zeros(img_gray.cols, img_gray.rows, CV_8UC1));
			}
			GpuHoG gpuHog;
			gpuHog.compute(img_gray, vImgHog);

		}
	}
	{
		boost::progress_timer t;

		HoG hog;
		for (int i = 0; i < 100; ++i) {
			/* code */

			vector<Mat> vImgHog;
			for (int j = 0; j < 9; ++j) {
				vImgHog.push_back(Mat::zeros(img_gray.cols, img_gray.rows, CV_8UC1));
			}
			Mat I_x, I_y;
			Mat orient = Mat::zeros(img_gray.cols, img_gray.rows, CV_8UC1);
			Mat mag = Mat::zeros(img_gray.cols, img_gray.rows, CV_8UC1);

			// |I_x|, |I_y|
			Sobel(img_gray, I_x, CV_16UC1, 1, 0, 3);
			Sobel(img_gray, I_y, CV_16UC1, 0, 1, 3);

			{
				// Orientation of gradients
				for (int y = 0; y < img.rows; ++y) {
					short *dataX = I_x.ptr<short>(y);
					short *dataY = I_y.ptr<short>(y);
					uchar *dataZ = orient.ptr<uchar>(y);

					for (int x = 0; x < img.cols; ++x) {
						// Avoid division by zero
						float tx = dataX[x] + copysign(0.000001f, (float)dataX[x]);
						// Scaling [-pi/2 pi/2] -> [0 80*pi]
						dataZ[x] = uchar( ( atan((float)dataY[x] / tx) + 3.14159265f / 2.0f ) * 80 );
					}
				}
			}


			{
				// Magnitude of gradients
				for (int y = 0; y < img.rows; ++y) {
					short *dataX = I_x.ptr<short>(y);
					short *dataY = I_y.ptr<short>(y);
					uchar *dataZ = mag.ptr<uchar>(y);

					for (int x = 0; x < img.cols; ++x) {
						dataZ[x] = (uchar)( sqrt(float(dataX[x] * dataX[x] + dataY[x] * dataY[x])) );
					}
				}
			}

			// 9-bin HOG feature stored at vImg[7] - vImg[15]
			hog.extractOBin(orient, mag, vImgHog);
		}
	}
	/*
	cout << "descr size: " << gpu_hog.getDescriptorSize() << endl;
	cout << "hist size: " << gpu_hog.getBlockHistogramSize() << endl;
	cout << "widht: " << hog_desc.cols << endl;
	cout << "height: " << hog_desc.rows << endl;
	cout << "channels: " << hog_desc.channels() << endl;
	*/


	/*Mat vis = get_hogdescriptor_visual_image(img, hog_desc, Size(320, 240), Size(8, 8), 2, 2);
	imshow("vis", vis);
	waitKey(0);*/
}

int main(int argc, char const *argv[]) {

	if (argc > 1)
		src_file = argv[1];

	Mat src_mat = imread(src_file);
	if (!src_mat.data) {
		cerr << "could not open image " << src_file << endl;
	}


	gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
	hogIt(src_mat);

}