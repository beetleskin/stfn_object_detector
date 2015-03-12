#include <Eigen/Eigenvalues> 
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using namespace std;
using namespace cv;

typedef Eigen::Matrix< float, 5, 1 > Vector5f;


static const float MM_PER_M = 1000.;
static const float M_PER_MM = 1.0/MM_PER_M;
static const float F_X = 570.3;
static const float F_Y = 570.3;



inline void calcPC(Mat &normals, Mat &points, Mat &depth_img, Mat &pc, int k=5, float max_dist=0.02, bool dist_rel_z=true) {

	if (pc.rows != depth_img.rows || pc.cols != depth_img.cols || pc.channels() != 5) {
		pc = Mat::zeros(depth_img.rows, depth_img.cols, CV_32FC(5));
	}
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver;
	Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
	int failed = 0;

	for (int y = 0; y < depth_img.rows; ++y) {
		for (int x = 0; x < depth_img.cols; ++x) {

			Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
			Eigen::Vector3f _m = Eigen::Vector3f::Zero();
			Eigen::Vector3f n_q = normals.at<Eigen::Vector3f>(y,x);
			Eigen::Vector3f p_q = points.at<Eigen::Vector3f>(y,x);
			std::vector<Eigen::Vector3f> m_j_list;
			Eigen::Matrix3f M = (I - n_q*(n_q.transpose()));
			float max_dist_rel = max_dist * ((dist_rel_z)? p_q[2]*1.5 : 1);


			for (int k_y = y-k/2; k_y <= y+k/2; ++k_y) {
				for (int k_x = x-k/2; k_x <= x+k/2; ++k_x) {

					if(k_y<0 || k_x<0 || k_y>=depth_img.rows || k_x >= depth_img.cols)
						continue;
					if(depth_img.at<float>(k_y,k_x) == 0)
						continue;

					Eigen::Vector3f p_j = points.at<Eigen::Vector3f>(k_y,k_x);

					if( max_dist_rel <= 0 || ((p_q - p_j).norm() < max_dist_rel) ) {
						Eigen::Vector3f n_j = normals.at<Eigen::Vector3f>(k_y,k_x);
						Eigen::Vector3f m_j = M * n_j;
						m_j_list.push_back(m_j);
						_m += m_j;
					}
					
				}
			}

			if(m_j_list.size() >= k) {
				_m /= m_j_list.size();
				for (int i = 0; i < m_j_list.size(); ++i) {
					A += (m_j_list[i] - _m)*((m_j_list[i] - _m).transpose());
				}
				A /= m_j_list.size();
				solver.computeDirect(A);
				float diff = solver.eigenvalues()(2) - solver.eigenvalues()(1);
				float mean = (solver.eigenvalues()(2) + solver.eigenvalues()(1)) / 2;
				float ratio = solver.eigenvalues()(1) / solver.eigenvalues()(2);
				Eigen::Vector3f evec = solver.eigenvectors().col(2);
				pc.at<Vector5f>(y,x) = Vector5f();
				pc.at<Vector5f>(y,x) << 
					solver.eigenvalues()(1),
					solver.eigenvalues()(2),
					evec;
			} else {
				failed++;
				pc.at<Vector5f>(y,x) = Vector5f::Zero();
				pc.at<Vector5f>(y,x) << std::numeric_limits<float>::quiet_NaN(),
										std::numeric_limits<float>::quiet_NaN(),
										std::numeric_limits<float>::quiet_NaN(),
										std::numeric_limits<float>::quiet_NaN(),
										std::numeric_limits<float>::quiet_NaN();
			}
		}
	}
}


inline bool calcPointsPCL(Mat &depth_img, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, float scale) {

	// TODO: dont handle only scale, but also the offset (c_x, c_y) of the given images center to the original image center (for training and roi images!)
	
	cloud.reset(new pcl::PointCloud<pcl::PointXYZ>(depth_img.cols, depth_img.rows));
	const float bad_point = 0;//std::numeric_limits<float>::quiet_NaN ();
	const float constant_x = M_PER_MM / F_X;
	const float constant_y = M_PER_MM / F_Y;
	bool is_valid = false;
	int centerX = depth_img.cols/2.0;
	int centerY = depth_img.rows/2.0;
	float x, y, z;
	int row, col = 0;

	for (row = 0, y = -centerY; row < depth_img.rows; ++row, ++y) {
		float* r_ptr = depth_img.ptr<float>(row);

		for (col = 0, x = -centerX; col < depth_img.cols; ++col, ++x) {
			pcl::PointXYZ newPoint;
			z = r_ptr[col];

			if(z) {
				newPoint.x = (x/scale)*z*constant_x;
				newPoint.y = (y/scale)*z*constant_y;
				newPoint.z = z*M_PER_MM;
				is_valid = true;
			} else {
				newPoint.x = newPoint.y = newPoint.z = bad_point;
			}
			cloud->at(col,row) = newPoint;
		}
	}

	return is_valid;
}


inline bool calcPointsRGBPCL(Mat &depth_img, Mat &bgr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &cloud, float scale) {

	// TODO: dont handle only scale, but also the offset (c_x, c_y) of the given images center to the original image center (for training and roi images!)
	cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>(depth_img.cols, depth_img.rows));
	const float bad_point = std::numeric_limits<float>::quiet_NaN ();
	const float constant_x = M_PER_MM / F_X;
	const float constant_y = M_PER_MM / F_Y;
	bool is_valid = false;
	float centerX = depth_img.cols/2.0;
	float centerY = depth_img.rows/2.0;
	float x, y, z;
	int row, col = 0;

	for (row = 0, y = -centerY; row < depth_img.rows; ++row, ++y) {
		float* r_ptr_depth = depth_img.ptr<float>(row);
		Vec3b* r_ptr_bgr = bgr.ptr<Vec3b>(row);

		for (col = 0, x = -centerX; col < depth_img.cols; ++col, ++x) {
			pcl::PointXYZRGB newPoint(r_ptr_bgr[col][2], r_ptr_bgr[col][1], r_ptr_bgr[col][0]);
			z = r_ptr_depth[col];

			if(z) {
				newPoint.x = (x/scale)*z*constant_x;
				newPoint.y = (y/scale)*z*constant_y;
				newPoint.z = z*M_PER_MM;
				is_valid = true;
			} else {
				newPoint.x = newPoint.y = newPoint.z = bad_point;
			}
			cloud->at(col,row) = newPoint;
		}
	}

	return is_valid;
}




inline void calcPoints(Mat &depth_img, Mat &points, float scale=1.0, int center_offset_x=0, int center_offset_y=0) {

	if (points.rows != depth_img.rows || points.cols != depth_img.cols || points.type() != CV_32FC3) {
		points = cv::Mat::zeros(depth_img.rows, depth_img.cols, CV_32FC3);
	}

	const float bad_point = 0.0f;
	const float constant_x = M_PER_MM / F_X;
	const float constant_y = M_PER_MM / F_Y;
	int centerX = depth_img.cols/2.0;
	int centerY = depth_img.rows/2.0;
	float x, y, z;
	int row, col = 0;

	for (row = 0, y = -centerY+center_offset_y; row < depth_img.rows; ++row, ++y) {
		float* r_ptr_src = depth_img.ptr<float>(row);
		Eigen::Vector3f* r_ptr_dst = points.ptr<Eigen::Vector3f>(row);

		for (col = 0, x = -centerX+center_offset_x; col < depth_img.cols; ++col, ++x) {
			Eigen::Vector3f &newPoint = r_ptr_dst[col];
			z = r_ptr_src[col];

			if(z) {
				newPoint[0] = (x/scale)*z*constant_x;
				newPoint[1] = (y/scale)*z*constant_y;
				newPoint[2] = z*M_PER_MM;
			} else {
				newPoint[0] = newPoint[1] = newPoint[2] = bad_point;
			}
		}
		/*
		float* r_ptr_src = depth_img.ptr<float>(row);
		float* r_ptr_dst = points.ptr<float>(row);

		for (col = 0, x = -centerX; col < depth_img.cols; ++col, ++x) {
			z = r_ptr_src[col];

			if(z) {
				r_ptr_dst[col*3] = (x/scale)*z*constant_x;
				r_ptr_dst[col*3+1] = (y/scale)*z*constant_x;
				r_ptr_dst[col*3+2] = z*M_PER_MM;
			} else {
				r_ptr_dst[col*3] = r_ptr_dst[col*3+1] = r_ptr_dst[col*3+2] = bad_point;
			}

		}*/
	}
}


inline void calcNormalsEigen(Mat &depth_img, Mat &points, Mat &normals, int k=11, float max_dist=0.02, bool dist_rel_z=true) {

	if (normals.rows != depth_img.rows || normals.cols != depth_img.cols || normals.channels() != 3) {
		normals = cv::Mat::zeros(depth_img.rows, depth_img.cols, CV_32FC3);
	}
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver;
	const float bad_point = std::numeric_limits<float>::quiet_NaN ();

	for (int y = 0; y < depth_img.rows; ++y) {
		for (int x = 0; x < depth_img.cols; ++x) {

			Eigen::Vector3f p_q = points.at<Eigen::Vector3f>(y,x);
			// depth-nan handle: bad point
			if (depth_img.at<float>(y, x) == 0 || p_q(0) != p_q(0)){
				normals.at<Eigen::Vector3f>(y,x) = Eigen::Vector3f(bad_point, bad_point, bad_point);
				continue;
			}

			Eigen::Matrix3f A = Eigen::Matrix3f::Zero();
			std::vector<Eigen::Vector3f> p_j_list;
			Eigen::Vector3f _p = Eigen::Vector3f::Zero();
			float max_dist_rel = max_dist * ((dist_rel_z)? p_q[2]*1.5 : 1);
			
			for (int k_y = y-k/2; k_y <= y+k/2; ++k_y) {
				for (int k_x = x-k/2; k_x <= x+k/2; ++k_x) {

					if(k_y<0 || k_x<0 || k_y>=depth_img.rows || k_x >= depth_img.cols)
						continue;
					if (k_y == y && k_x == x)
						continue;
					if (depth_img.at<float>(k_y, k_x) == 0)
						continue;

					Eigen::Vector3f p_j = points.at<Eigen::Vector3f>(k_y,k_x);
					if( max_dist_rel <= 0 || ((p_q - p_j).norm() <= max_dist_rel) ) {
						p_j_list.push_back(p_j);
						_p += p_j;
					}
				}
			}


			_p /= p_j_list.size();
			double weight_sum = 0;
			for (int i = 0; i < p_j_list.size(); ++i) {
				double w = 1.0/(p_j_list[i] - _p).squaredNorm();
				A += w*((p_j_list[i] - _p)*((p_j_list[i] - _p).transpose()));
				weight_sum += w;
			}
			A /= weight_sum;
			solver.computeDirect(A);
			Eigen::Vector3f normal = solver.eigenvectors().col(0).normalized();
			// flip to viewpoint (0,0,0)
			if(normal(2) > 0)
				normal *= -1;
			normals.at<Eigen::Vector3f>(y,x) = normal;
		}
	}
}


inline void calcNormalsGrad(Mat &dx, Mat &dy, Mat &normals) {

	Mat normalsLength(dx.rows, dx.cols, CV_32FC1);
	Mat dz = Mat::ones(dx.rows, dx.cols, CV_32FC1);

	vector<Mat> channels(3);
	channels[0] = -dx.clone();
	channels[1] = -dy.clone();
	channels[2] = dz;

	// normalize the shit out of it
	sqrt(channels[0].mul(channels[0]) + channels[1].mul(channels[1]) + channels[2].mul(channels[2]), normalsLength);
	channels[0] /= normalsLength;
	channels[1] /= normalsLength;
	channels[2] /= normalsLength;

	merge(channels, normals);
}


inline void recursiveMedianFilter(Mat &depth_img, Mat& depth_shadow_dist) {
	Mat tmp;

	/* create depth_shadow_dist: distance of each pixel to the next non-shadow 
	 * (non-black, valid) pixel as an inverse credebility measure (0 is good, increasing
	 * value is increasingly bad).
	 */
	threshold(depth_img, tmp, 0, 255, CV_THRESH_BINARY);
	tmp = 255 - tmp;
	tmp.convertTo(tmp, CV_8UC1);
	cv::distanceTransform(tmp, depth_shadow_dist, CV_DIST_L2, 3);
	

	// gather all zero-pixel positions
	vector<cv::Point> zeroIdx;
	for(int y = 0; y< depth_img.rows; y++) {
		for(int x = 0; x< depth_img.cols; x++) {
			if(depth_img.at<float>(y,x) == 0){
				zeroIdx.push_back(cv::Point(x,y));
			}
		}
	}

	int total_zeros = zeroIdx.size();
	int oldSize = 0;
	int num_dist_removed_zeros = 0;
	// max distance to a non-black pixel
	float depth_shadow_dist_max = (float)(depth_img.rows + depth_img.cols) / 20.0;
	

	while(!zeroIdx.empty() && oldSize != zeroIdx.size()) {

		//TODO: sort by distance map values!
		oldSize = zeroIdx.size();
		depth_img.copyTo(tmp);


		for(int i = 0; i < zeroIdx.size(); i++) {
			if(depth_shadow_dist.at<float>(zeroIdx[i].y, zeroIdx[i].x) > depth_shadow_dist_max) {
				num_dist_removed_zeros++;
				zeroIdx.erase(zeroIdx.begin() + i);
				i--;
				continue;
			}

			int y1 = (zeroIdx[i].y-2 >= 0)? zeroIdx[i].y-2 : 0;
			int x1 = (zeroIdx[i].x-2 >= 0)? zeroIdx[i].x-2 : 0;
			int y2 = (zeroIdx[i].y+3 < depth_img.rows)? zeroIdx[i].y+3 : depth_img.rows -1;
			int x2 = (zeroIdx[i].x+3 < depth_img.cols)? zeroIdx[i].x+3 : depth_img.cols -1;

			cv::Rect roiRect(x1,y1,x2-x1,y2-y1);
			cv::Mat roi(depth_img, cv::Rect(x1,y1,x2-x1,y2-y1));
			
			int countNonZero = 0;
			vector<float> values;
			for(int y = 0; y < roi.rows; y++) {
				for(int x = 0; x < roi.cols; x++) {
					if(roi.at<float>(y,x) > 0) {
						countNonZero++;
						values.push_back(roi.at<float>(y,x));
					}
				}
			}

			if(countNonZero > 0) {
				std::sort(values.begin(), values.end());
      			float median = values[values.size() / 2];
				tmp.at<float>(zeroIdx[i].y, zeroIdx[i].x) = median;
				zeroIdx.erase(zeroIdx.begin() + i);
				i--;
			}

		}
		tmp.copyTo(depth_img);
	}
}