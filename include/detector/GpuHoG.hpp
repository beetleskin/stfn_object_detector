#include <boost/assert.hpp>
#include <assert.h>

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;
using namespace cv;



class GpuHoG {
public:
	GpuHoG() {
		this->blockSize = Size(16, 16);
		this->blockStride = Size(8, 8);
		this->cellSize = Size(8, 8);
		this->nBins = 9;
		this->win_sigma = gpu::HOGDescriptor::DEFAULT_WIN_SIGMA;
		this->threshold_L2hys = 0.2;
		this->gamma_correction = true;
		this->nLevels = gpu::HOGDescriptor::DEFAULT_NLEVELS;
		this->descr_format = gpu::HOGDescriptor::DESCR_FORMAT_COL_BY_COL;
	}

	void compute(gpu::GpuMat &gpu_img_gray, vector<gpu::GpuMat> &gpu_vImgHog) {
		BOOST_ASSERT_MSG(gpu_vImgHog.size() == this->nBins, "size of vImgHog needs to be GpuHoG::nBins");

		// check if image size is compatible with the gpu hog
		gpu::GpuMat gpu_img_gray_border_converted;
		Size winSize(gpu_img_gray.cols, gpu_img_gray.rows);
		Size winSizeBlockMod(
		    (winSize.width - this->blockSize.width ) % this->blockSize.width,
		    (winSize.height - this->blockSize.height ) % this->blockSize.height
		);
		Size blockLeftover(0, 0);
		if (winSizeBlockMod.width != 0 || winSizeBlockMod.height != 0) {
			blockLeftover = this->blockSize - winSizeBlockMod;
			gpu::copyMakeBorder(gpu_img_gray, gpu_img_gray_border_converted, 0, blockLeftover.height, 0, blockLeftover.width, BORDER_REPLICATE);
			winSize = gpu_img_gray_border_converted.size();
		}


		// compute HoG descriptor
		gpu::GpuMat gpu_hog_descr;
		gpu::HOGDescriptor gpu_hog(winSize, this->blockSize, this->blockStride, this->cellSize, this->nBins, this->win_sigma, this->threshold_L2hys, this->gamma_correction, this->nLevels);
		
		if (blockLeftover.width != 0 || blockLeftover.height != 0) {
			gpu_hog.getDescriptors(gpu_img_gray_border_converted, this->blockSize, gpu_hog_descr, this->descr_format);
		} else {
			gpu_hog.getDescriptors(gpu_img_gray, this->blockSize, gpu_hog_descr, this->descr_format);

		}
		
		// copy data from gpu to ram
		Mat hog_desc(gpu_hog_descr);

		// release gpu data
		gpu_hog_descr.release();

		// convert data
		float *descriptorDataIdx = hog_desc.ptr<float>();
		int cells_in_x_dir = winSize.width / cellSize.width;
		int cells_in_y_dir = winSize.height / cellSize.height;
		int blocks_in_x_dir = cells_in_x_dir - 1;
		int blocks_in_y_dir = cells_in_y_dir - 1;

		Mat counter = Mat::zeros(cells_in_y_dir, cells_in_x_dir, CV_32FC1);
		vector<Mat> vImgHogTmp;
		for (int i = 0; i < this->nBins; ++i) {
			vImgHogTmp.push_back(Mat::zeros(cells_in_y_dir, cells_in_x_dir, CV_32FC1));
		}

		for (int blockx = 0; blockx < blocks_in_x_dir; ++blockx) {
			for (int blocky = 0; blocky < blocks_in_y_dir; ++blocky) {
				for (int cellNr = 0; cellNr < 4; ++cellNr) {
					int cellx = blockx;
					int celly = blocky;
					if (cellNr == 1) celly++;
					if (cellNr == 2) cellx++;
					if (cellNr == 3) {
						cellx++;
						celly++;
					}

					counter.at<float>(celly, cellx) += 1;
					for (int bin = 0; bin < this->nBins; ++bin) {
						vImgHogTmp[bin].ptr<float>(celly)[cellx] += *descriptorDataIdx;
						descriptorDataIdx++;

					}
				}
			}
		}

		gpu::Stream gpu_stream;
		Mat tmp(winSize, CV_8UC1);
		for (int bin = 0; bin < this->nBins; ++bin) {
			vImgHogTmp[bin] /= counter;
			resize(vImgHogTmp[bin], vImgHogTmp[bin], winSize);

			if (blockLeftover.width != 0 || blockLeftover.height != 0) {
				Rect originalImgRoi(0, 0, winSize.width - blockLeftover.width, winSize.height - blockLeftover.height);
				convertScaleAbs(vImgHogTmp[bin](originalImgRoi), tmp, 255);
			} else {
				convertScaleAbs(vImgHogTmp[bin], tmp, 255);
			}

			// upload
			gpu_stream.enqueueUpload(tmp, gpu_vImgHog[bin]);
		}
		gpu_stream.waitForCompletion();
	}

	void compute(Mat &img_gray, vector<Mat> &vImgHog) {
		BOOST_ASSERT_MSG(vImgHog.size() == this->nBins, "size of vImgHog needs to be GpuHoG::nBins");

		// check if image size is compatible with the gpu hog
		Mat img_gray_border_converted;
		Size winSize(img_gray.cols, img_gray.rows);
		Size winSizeBlockMod(
		    (winSize.width - this->blockSize.width ) % this->blockSize.width,
		    (winSize.height - this->blockSize.height ) % this->blockSize.height
		);
		Size blockLeftover(0, 0);
		if (winSizeBlockMod.width != 0 || winSizeBlockMod.height != 0) {
			blockLeftover = this->blockSize - winSizeBlockMod;
			copyMakeBorder(img_gray, img_gray_border_converted, 0, blockLeftover.height, 0, blockLeftover.width, BORDER_REPLICATE);
			winSize = img_gray_border_converted.size();
		}


		// upload data to gpu
		gpu::GpuMat gpu_img_gray;
		if (blockLeftover.width != 0 || blockLeftover.height != 0) {
			gpu_img_gray.upload(img_gray_border_converted);
		} else {
			gpu_img_gray.upload(img_gray);
		}

		// compute HoG descriptor
		gpu::GpuMat gpu_hog_descr;
		gpu::HOGDescriptor gpu_hog(winSize, this->blockSize, this->blockStride, this->cellSize, this->nBins, this->win_sigma, this->threshold_L2hys, this->gamma_correction, this->nLevels);
		gpu_hog.getDescriptors(gpu_img_gray, this->blockSize, gpu_hog_descr, this->descr_format);

		// copy data from gpu to ram
		Mat hog_desc(gpu_hog_descr);

		// release gpu data
		gpu_img_gray.release();
		gpu_hog_descr.release();

		// convert data
		float *descriptorDataIdx = hog_desc.ptr<float>();
		int cells_in_x_dir = winSize.width / cellSize.width;
		int cells_in_y_dir = winSize.height / cellSize.height;
		int blocks_in_x_dir = cells_in_x_dir - 1;
		int blocks_in_y_dir = cells_in_y_dir - 1;

		Mat counter = Mat::zeros(cells_in_y_dir, cells_in_x_dir, CV_32FC1);
		vector<Mat> vImgHogTmp;
		for (int i = 0; i < this->nBins; ++i) {
			vImgHogTmp.push_back(Mat::zeros(cells_in_y_dir, cells_in_x_dir, CV_32FC1));
		}

		for (int blockx = 0; blockx < blocks_in_x_dir; ++blockx) {
			for (int blocky = 0; blocky < blocks_in_y_dir; ++blocky) {
				for (int cellNr = 0; cellNr < 4; ++cellNr) {
					int cellx = blockx;
					int celly = blocky;
					if (cellNr == 1) celly++;
					if (cellNr == 2) cellx++;
					if (cellNr == 3) {
						cellx++;
						celly++;
					}

					counter.at<float>(celly, cellx) += 1;
					for (int bin = 0; bin < this->nBins; ++bin) {
						vImgHogTmp[bin].ptr<float>(celly)[cellx] += *descriptorDataIdx;
						descriptorDataIdx++;

					}
				}
			}
		}

		Mat tmp(winSize, CV_32FC1);
		for (int bin = 0; bin < this->nBins; ++bin) {
			vImgHogTmp[bin] /= counter;
			resize(vImgHogTmp[bin], tmp, winSize);

			if (blockLeftover.width != 0 || blockLeftover.height != 0) {
				Rect originalImgRoi(0, 0, winSize.width - blockLeftover.width, winSize.height - blockLeftover.height);
				convertScaleAbs(tmp(originalImgRoi), vImgHog[bin], 255);
			} else {
				convertScaleAbs(tmp, vImgHog[bin], 255);
			}
		}
	}

private:
	Size blockSize;
	Size cellSize;
	Size blockStride;
	int nBins;
	double win_sigma;
	double threshold_L2hys;
	bool gamma_correction;
	int nLevels;
	int descr_format;
};