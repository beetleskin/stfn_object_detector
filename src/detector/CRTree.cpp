#include "CRTree.hpp"
#include "TrainStats.hpp"

#include <fstream>


using namespace std;


unsigned int CRTree::treeCount = 0;


// Read tree from file
CRTree::CRTree(const char *filename, bool &success) {
	cout << "Load Tree " << filename << endl;

	int dummy;

	ifstream in(filename);
	success = true;
	if (in.is_open()) {
		// get the scale of the tree
		in >> dummy;
		in >> max_depth;

		in >> num_nodes;
		nodes.resize(num_nodes);

		in >> num_leaf;
		leafs.resize(num_leaf);

		in >> num_labels;

		// class structure
		class_id = new int[num_labels];
		for (unsigned int n = 0; n < num_labels; ++n)
			in >> class_id[n];

		int node_id;
		int isLeaf;
		// read tree nodes
		for (unsigned int n = 0; n < num_nodes; ++n) {
			in >> node_id;
			nodes[node_id].idN = node_id;
			in >> nodes[node_id].depth;
			in >> nodes[node_id].isLeaf;
			in >> nodes[node_id].parent;
			in >> nodes[node_id].leftChild;
			in >> nodes[node_id].rightChild;
			nodes[node_id].data.resize(6);
			for (unsigned int i = 0; i < 6; ++i) {
				in >> nodes[node_id].data[i];
			}
		}

		// read tree leafs
		LeafNode *ptLN;
		for (unsigned int l = 0; l < num_leaf; ++l) {
			ptLN = &leafs[l];
			in >> ptLN->idL;
			in >> ptLN->cL;
			in >> ptLN->eL;
			in >> ptLN->fL;
			ptLN->vPrLabel.resize(num_labels);
			ptLN->vCenter.resize(num_labels);
			ptLN->vCenterWeights.resize(num_labels);
			ptLN->vID.resize(num_labels);
			ptLN->vLabelDistrib.resize(num_labels);
			ptLN->nOcc.resize(num_labels);

			for (unsigned int c = 0; c < num_labels; ++c) {
				in >> ptLN->vPrLabel[c];
				in >> dummy;
				in >> ptLN->vLabelDistrib[c];

				if (ptLN->vPrLabel[c] < 0) {
					std::cerr << ptLN->vPrLabel[c] << std::endl;
				}

				ptLN->vCenter[c].resize(dummy);
				ptLN->vCenterWeights[c].resize(dummy);
				ptLN->vID[c].resize(dummy);
				ptLN->nOcc[c] = dummy;

				float temp_weight = 1.0f / float(ptLN->nOcc[c]);

				for (int i = 0; i < dummy; ++i) {
					in >> ptLN->vCenter[c][i].x;
					in >> ptLN->vCenter[c][i].y;
					ptLN->vCenterWeights[c][i] = temp_weight;
					in >> ptLN->vID[c][i];
				}
			}
		}

	} else {
		success = false;
		cerr << "Could not read tree: " << filename << endl;
	}

	in.close();
}

/////////////////////// IO Function /////////////////////////////
bool CRTree::saveTree(const char *filename) const {
	bool done = false;

	ofstream out(filename);
	if (out.is_open()) {

		out << 1.0 << " " << max_depth << " " << num_nodes << " " << num_leaf << " " << num_labels << endl;

		// store class structure
		for (unsigned int n = 0; n < num_labels; ++n)
			out << class_id[n] << " ";
		out << endl;

		// save tree nodes
		for (unsigned int n = 0; n < num_nodes; ++n) {

			out << nodes[n].idN;
			out << " " << nodes[n].depth;
			out << " " << nodes[n].isLeaf;
			out << " " << nodes[n].parent;
			out << " " << nodes[n].leftChild;
			out << " " << nodes[n].rightChild;

			for (unsigned int i = 0; i < 6; ++i) {
				out << " " << nodes[n].data[i];
			}
			out << endl;
		}
		out << endl;

		// save tree leaves
		for (unsigned int l = 0; l < num_leaf; ++l) {
			const LeafNode *ptLN = &leafs[l];
			out << ptLN->idL << " ";
			out << ptLN->cL << " ";
			out << ptLN->eL << " ";
			out << ptLN->fL << " ";

			for (unsigned int c = 0; c < num_labels; ++c) {
				out << ptLN->vPrLabel[c] << " " << ptLN->vCenter[c].size() << " " << ptLN->vLabelDistrib[c] << " ";

				for (unsigned int i = 0; i < ptLN->vCenter[c].size(); ++i) {
					out << ptLN->vCenter[c][i].x << " " << ptLN->vCenter[c][i].y << " " << ptLN->vID[c][i] << " ";
				}
			}
			out << endl;
		}

		out.close();

		done = true;
	}

	return done;
}

bool CRTree::loadHierarchy(const char *filename) {
	ifstream in(filename);
	int number_of_nodes = 0;
	if (in.is_open()) {
		in >> number_of_nodes;
		hierarchy.resize(number_of_nodes);
		int temp;
		for (int nNr = 0; nNr < number_of_nodes; nNr++) {
			in >> hierarchy[nNr].id;
			in >> hierarchy[nNr].leftChild;
			in >> hierarchy[nNr].rightChild;
			in >> hierarchy[nNr].linkage;
			in >> hierarchy[nNr].parent;
			in >> temp;
			hierarchy[nNr].subclasses.resize(temp);
			for (int sNr = 0; sNr < temp; sNr++)
				in >> hierarchy[nNr].subclasses[sNr];
		}
		in.close();
		return true;
	} else {
		std::cerr << " failed to read the hierarchy file: " << filename << std::endl;
		return false;
	}
}

/////////////////////// Training Function /////////////////////////////

// Start grow tree
void CRTree::growTree(const CRPatch &TrData, int samples) {
	// Get inverse numbers of patches
	vector<float> vRatio(TrData.vLPatches.size());

	vector < vector<const PatchFeature *> > TrainSet(TrData.vLPatches.size());
	vector < vector<int> > TrainIDs(TrData.vImageIDs.size());

	for (unsigned int l = 0; l < TrainSet.size(); ++l) {
		TrainSet[l].resize(TrData.vLPatches[l].size());
		TrainIDs[l].resize(TrData.vImageIDs[l].size());

		if (TrainSet[l].size() > 0) {
			vRatio[l] = 1.0f / (float) TrainSet[l].size();
		} else {
			vRatio[l] = 0.0f;
		}
		for (unsigned int i = 0; i < TrainSet[l].size(); ++i) {
			TrainSet[l][i] = &TrData.vLPatches[l][i];
		}
		for (unsigned int i = 0; i < TrainIDs[l].size(); ++i) {
			TrainIDs[l][i] = TrData.vImageIDs[l][i];
		}
	}
	// Grow tree
	grow(TrainSet, TrainIDs, 0, 0, samples, vRatio);
}

// Called by growTree
void CRTree::grow(const vector<vector<const PatchFeature *> > &TrainSet, const vector<vector<int> > &TrainIDs, int node, unsigned int depth, int samples, vector<float> &vRatio) {

	if (depth < max_depth) {
		vector < vector<const PatchFeature *> > SetA;
		vector < vector<const PatchFeature *> > SetB;
		vector < vector<int> > idA;
		vector < vector<int> > idB;

		int test[6];

		// Set measure mode for split: -1 - classification, otherwise - regression (for locations)
		int stat[TrainSet.size()];
		int count_stat = getStatSet(TrainSet, stat);

		bool check_test = false;
		int count_test = 0;

		while (!check_test) {
			int measure_mode = 0;
			if (count_stat > 1)
				measure_mode = (cvRandInt(cvRNG) % 2) - 1;

			// Find optimal test
			if (check_test = optimizeTest(SetA, SetB, idA, idB, TrainSet, TrainIDs, test, samples, measure_mode, vRatio)) {

				TrainStats::get().addSplit(this->id, node, measure_mode);


				// Store binary test for current node
				InternalNode *ptT = &nodes[node];
				ptT->data.resize(6);
				for (int t = 0; t < 6; ++t)
					ptT->data[t] = test[t];

				double countA = 0;
				double countB = 0;
				for (unsigned int l = 0; l < TrainSet.size(); ++l) {
					countA += SetA[l].size();
					countB += SetB[l].size();
				}

				//make an empty node and push it to the tree
				InternalNode temp;
				temp.rightChild = -1;
				temp.leftChild = -1;
				temp.parent = node;
				temp.data.resize(6, 0);
				temp.depth = depth + 1;

				// Go left
				temp.idN = nodes.size();
				nodes[node].leftChild = temp.idN;

				// If enough patches are left continue growing else stop
				if (countA > min_samples) {
					temp.isLeaf = false;
					nodes.push_back(temp);
					num_nodes += 1;
					grow(SetA, idA, temp.idN, depth + 1, samples, vRatio);
				} else {
					// the leaf id will be assigned to the left child in the makeLeaf
					// isLeaf will be set to true
					temp.isLeaf = true;
					nodes.push_back(temp);
					num_nodes += 1;
					makeLeaf(SetA, idA, vRatio, temp.idN);
				}

				// Go right
				temp.idN = nodes.size();
				nodes[node].rightChild = temp.idN;
				// If enough patches are left continue growing else stop
				if (countB > min_samples) {
					temp.isLeaf = false;
					nodes.push_back(temp);
					num_nodes += 1;
					grow(SetB, idB, temp.idN, depth + 1, samples, vRatio);
				} else {
					temp.isLeaf = true;
					nodes.push_back(temp);
					num_nodes += 1;
					makeLeaf(SetB, idB, vRatio, temp.idN);
				}

			} else {

				if (++count_test > 3) {

					TrainStats::get().addInvalidTest(this->id, node);

					// Could not find split (only invalid splits)
					nodes[node].isLeaf = true;
					nodes[node].leftChild = -1;
					nodes[node].rightChild = -1;
					nodes[node].data.resize(6, 0);
					makeLeaf(TrainSet, TrainIDs, vRatio, node);

					check_test = true;
				}
			}
		}
	} else {
		// maximum depth is reached
		nodes[node].isLeaf = true;
		nodes[node].leftChild = -1;
		nodes[node].rightChild = -1;
		nodes[node].data.resize(6, 0);
		// do not change the parent
		makeLeaf(TrainSet, TrainIDs, vRatio, node);
	}

}

// Create leaf node from patches
void CRTree::makeLeaf(const std::vector<std::vector<const PatchFeature *> > &TrainSet, const std::vector<std::vector<int> > &TrainIDs, std::vector<float> &vRatio, int node) {
	// setting the leaf pointer
	nodes[node].leftChild = num_leaf;
	LeafNode L;
	L.idL = num_leaf;
	L.vCenter.resize(TrainSet.size());
	L.vPrLabel.resize(TrainSet.size());
	L.vID.resize(TrainSet.size());

	// Store data
	float invsum = 0;
	float invsum_pos = 0;
	for (unsigned int l = 0; l < TrainSet.size(); ++l) {
		L.vPrLabel[l] = (float) TrainSet[l].size() * vRatio[l];
		invsum += L.vPrLabel[l];
		if (class_id[l] > 0) {
			invsum_pos += L.vPrLabel[l];
		}
		L.vCenter[l].resize(TrainSet[l].size());
		L.vID[l].resize(TrainIDs[l].size());
		for (unsigned int i = 0; i < TrainSet[l].size(); ++i) {
			L.vCenter[l][i] = TrainSet[l][i]->center;
			float depth_scale = TrainSet[l][i]->vPatch[depth_channel].ptr<float>(10)[10];
			if (depth_scale  < 0.1)
				depth_scale = 1.0;
			L.vCenter[l][i].x = L.vCenter[l][i].x * depth_scale + 0.5;
			L.vCenter[l][i].y = L.vCenter[l][i].y * depth_scale + 0.5;

			L.vID[l][i] = TrainIDs[l][i];
		}
	}

	// Normalize probability
	invsum = 1.0f / invsum;
	if (invsum_pos > 0) {
		invsum_pos = 1.0f / invsum_pos;
		for (unsigned int l = 0; l < TrainSet.size(); ++l) {
			L.vPrLabel[l] *= invsum;
		}
		L.cL = invsum / invsum_pos;
	} else { // there is no positive patch in this leaf
		for (unsigned int l = 0; l < TrainSet.size(); ++l) {
			L.vPrLabel[l] *= invsum;
		}
		L.cL = 0.0f;
	}

	leafs.push_back(L);

	// Increase leaf counter
	++num_leaf;
}

bool CRTree::optimizeTest(vector<vector<const PatchFeature *> > &SetA, vector<vector<const PatchFeature *> > &SetB, vector<vector<int> > &idA, vector<vector<int> > &idB,
                          const vector<vector<const PatchFeature *> > &TrainSet, const vector<vector<int> > &TrainIDs, int *test, unsigned int iter, unsigned int measure_mode, const std::vector<float> &vRatio) {

	bool found = false;
	int subsample = 1000 * TrainSet.size();

	// sampling patches proportional to the class to keep the balance of the classes
	std::vector<int> subsample_perclass;
	subsample_perclass.resize(TrainSet.size(), 0);
	// first find out how many patches are there
	int all_patches = 0;
	for (int sz = 0; sz < TrainSet.size(); sz++)
		all_patches += TrainSet[sz].size();
	// the calculate the sampling rate for each set
	float sample_rate = float(subsample) / float(all_patches);
	for (int sz = 0; sz < TrainSet.size(); sz++) {
		subsample_perclass[sz] = int(sample_rate * float(TrainSet[sz].size()));
	}
	// now we can subsample the patches and their associated ids
	vector < vector<const PatchFeature *> > tmpTrainSet;
	vector < vector<int> > tmpTrainIDs;
	tmpTrainSet.resize(TrainSet.size());
	tmpTrainIDs.resize(TrainSet.size());

	// sample the patches in a regular grid and copy them to the tree
	for (int sz = 0; sz < TrainSet.size(); sz++) {
		tmpTrainSet[sz].resize(std::min(int(TrainSet[sz].size()), subsample_perclass[sz]));
		tmpTrainIDs[sz].resize(tmpTrainSet[sz].size());
		if (tmpTrainSet[sz].size() == 0)
			continue;

		float float_rate = float(TrainSet[sz].size()) / float(tmpTrainSet[sz].size());
		for (int j = 0; j < tmpTrainSet[sz].size(); j++) {
			tmpTrainSet[sz][j] = TrainSet[sz][int(float_rate * j)];
			tmpTrainIDs[sz][j] = TrainIDs[sz][int(float_rate * j)];
		}
	}

	double tmpDist;
	double bestDist = -DBL_MAX;
	int tmpTest[6];

	// find non-empty class
	int check_label = 0;
	while (check_label < (int) tmpTrainSet.size() && tmpTrainSet[check_label].size() == 0)
		++check_label;

	// Find best test of ITER iterations
	for (unsigned int i = 0; i < iter; ++i) {
		// temporary data for split into Set A and Set B
		vector < vector<const PatchFeature *> > tmpA(tmpTrainSet.size());
		vector < vector<const PatchFeature *> > tmpB(tmpTrainSet.size());
		vector < vector<int> > tmpIDA(tmpTrainIDs.size());
		vector < vector<int> > tmpIDB(tmpTrainIDs.size());
		// temporary data for finding best test
		vector < vector<IntIndex> > tmpValSet(tmpTrainSet.size());
		// generate binary test without threshold
		generateTest(&tmpTest[0], tmpTrainSet[check_label][0]->roi.width / 2, tmpTrainSet[check_label][0]->roi.height / 2, tmpTrainSet[check_label][0]->vPatch.size() - 1);

		// compute value for each patch
		evaluateTest(tmpValSet, &tmpTest[0], tmpTrainSet);

		// find min/max values for threshold
		int vmin = INT_MAX;
		int vmax = INT_MIN;
		for (unsigned int l = 0; l < tmpTrainSet.size(); ++l) {
			if (tmpValSet[l].size() > 0) {
				if (vmin > tmpValSet[l].front().val)
					vmin = tmpValSet[l].front().val;
				if (vmax < tmpValSet[l].back().val)
					vmax = tmpValSet[l].back().val;
			}
		}
		int d = vmax - vmin;

		if (d > 0) {

			// Find best threshold
			for (unsigned int j = 0; j < 10; ++j) {

				// Generate some random thresholds
				int tr = (cvRandInt(cvRNG) % (d)) + vmin;

				// Split training data into two sets A,B accroding to threshold t
				split(tmpA, tmpB, tmpIDA, tmpIDB, tmpTrainSet, tmpValSet, tmpTrainIDs, tr); // include idA , idB, TrainIDs
				int countA = 0;
				int countB = 0;
				for (unsigned int l = 0; l < tmpTrainSet.size(); ++l) {
					if (tmpA[l].size() > countA)
						countA = tmpA[l].size();
					if (tmpB[l].size() > countB)
						countB = tmpB[l].size();
				}

				// Do not allow empty set split (all patches end up in set A or B)

				if (countA > 10 && countB > 10) {
					// Measure quality of split with measure_mode 0 - classification, 1 - regression
					tmpDist = measureSet(tmpA, tmpB, measure_mode, vRatio);


					// Take binary test with best split
					if (tmpDist > bestDist) {

						found = true;
						bestDist = tmpDist;
						for (int t = 0; t < 5; ++t)
							test[t] = tmpTest[t];
						test[5] = tr;
					}
				}
			} // end for
			// - check if inf genereates a test, resp. survives (tmpDist > bestDist)
			// - check detection output: confidence value depending on number of scales?
			//TrainStats::get().addMeasure(this->id, tmpTest[4], measure_mode, bestDist);
		}
	} // end iter

	if (found) {
		// here we should evaluate the test on all the data
		vector < vector<IntIndex> > valSet(TrainSet.size());
		evaluateTest(valSet, &test[0], TrainSet);
		// now we can keep the best Test and split the whole set according to the best test and threshold
		SetA.resize(TrainSet.size());
		SetB.resize(TrainSet.size());
		idA.resize(TrainSet.size());
		idB.resize(TrainSet.size());
		split(SetA, SetB, idA, idB, TrainSet, valSet, TrainIDs, test[5]);
	}
	// return true if a valid test has been found
	// test is invalid if only splits with with members all less than 10 in set A or B has been created
	return found;
}

void CRTree::evaluateTest(std::vector<std::vector<IntIndex> > &valSet, const int *test, const std::vector<std::vector<const PatchFeature *> > &TrainSet) {
	for (unsigned int l = 0; l < TrainSet.size(); ++l) {
		valSet[l].resize(TrainSet[l].size());
		for (unsigned int i = 0; i < TrainSet[l].size(); ++i) {

			// pointer to channel
			const Mat &f_img = TrainSet[l][i]->vPatch[test[4]];
			float depth_scale = TrainSet[l][i]->vPatch[depth_channel].ptr<float>(10)[10];
			float x1 = test[0];
			float y1 = test[1];
			float x2 = test[2];
			float y2 = test[3];
			// scale
			if (depth_scale > 0.1) {
				x1 -= 5;
				y1 -= 5;
				x2 -= 5;
				y2 -= 5;
				x1 /= depth_scale;
				y1 /= depth_scale;
				x2 /= depth_scale;
				y2 /= depth_scale;
				x1 += 10;
				y1 += 10;
				x2 += 10;
				y2 += 10;
			} else {
				x1 += 5;
				y1 += 5;
				x2 += 5;
				y2 += 5;
			}

			if (x1 >= f_img.cols)
				x1 = f_img.cols - 1;
			if (y1 >= f_img.rows)
				y1 = f_img.rows - 1;
			if (x2 >= f_img.cols)
				x2 = f_img.cols - 1;
			if (y2 >= f_img.rows)
				y2 = f_img.rows - 1;
			if (x1 < 0)
				x1 = 0;
			if (y1 < 0)
				y1 = 0;
			if (x2 < 0)
				x2 = 0;
			if (y2 < 0)
				y2 = 0;
			int p1 = int(f_img.ptr<uchar>(int(y1 + 0.5f))[int(x1 + 0.5f)]);
			int p2 = int(f_img.ptr<uchar>(int(y2 + 0.5f))[int(x2 + 0.5f)]);

			valSet[l][i].val = p1 - p2;
			valSet[l][i].index = i;
		}
		sort(valSet[l].begin(), valSet[l].end());
	}
}

void CRTree::split(vector<vector<const PatchFeature *> > &SetA, vector<vector<const PatchFeature *> > &SetB, vector<vector<int> > &idA, vector<vector<int> > &idB,
                   const vector<vector<const PatchFeature *> > &TrainSet, const vector<vector<IntIndex> > &valSet, const vector<vector<int> > &TrainIDs, int t) {
	for (unsigned int l = 0; l < TrainSet.size(); ++l) {
		// search largest value such that val<t
		vector<IntIndex>::const_iterator it = valSet[l].begin();
		while (it != valSet[l].end() && it->val < t) {
			++it;
		}

		SetA[l].resize(it - valSet[l].begin());
		idA[l].resize(SetA[l].size());
		SetB[l].resize(TrainSet[l].size() - SetA[l].size());
		idB[l].resize(SetB[l].size());

		it = valSet[l].begin();
		for (unsigned int i = 0; i < SetA[l].size(); ++i, ++it) {
			SetA[l][i] = TrainSet[l][it->index];
			idA[l][i] = TrainIDs[l][it->index];
		}

		it = valSet[l].begin() + SetA[l].size();
		for (unsigned int i = 0; i < SetB[l].size(); ++i, ++it) {
			SetB[l][i] = TrainSet[l][it->index];
			idB[l][i] = TrainIDs[l][it->index];
		}

	}
}

// this code uses the class label!!!!
double CRTree::distMeanMC(const vector<vector<const PatchFeature *> > &SetA, const vector<vector<const PatchFeature *> > &SetB) {
	// calculating location entropy per class
	vector<double> meanAx(num_labels, 0);
	vector<double> meanAy(num_labels, 0);
	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			for (vector<const PatchFeature *>::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
				meanAx[c] += (*it)->center.x;
				meanAy[c] += (*it)->center.y;
			}
		}
	}

	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			meanAx[c] /= (double) SetA[c].size();
			meanAy[c] /= (double) SetA[c].size();
		}
	}

	vector<double> distA(num_labels, 0);
	int non_empty_classesA = 0;
	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			if (SetB[c].size() > 0)
				non_empty_classesA++;
			for (std::vector<const PatchFeature *>::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
				double tmp = (*it)->center.x - meanAx[c];
				distA[c] += tmp * tmp;
				tmp = (*it)->center.y - meanAy[c];
				distA[c] += tmp * tmp;
			}
		}
	}

	vector<double> meanBx(num_labels, 0);
	vector<double> meanBy(num_labels, 0);
	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			for (vector<const PatchFeature *>::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
				meanBx[c] += (*it)->center.x;
				meanBy[c] += (*it)->center.y;
			}
		}
	}

	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			meanBx[c] /= (double) SetB[c].size();
			meanBy[c] /= (double) SetB[c].size();
		}
	}

	vector<double> distB(num_labels, 0);
	int non_empty_classesB = 0;
	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			if (SetB[c].size() > 0)
				non_empty_classesB++;

			for (std::vector<const PatchFeature *>::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
				double tmp = (*it)->center.x - meanBx[c];
				distB[c] += tmp * tmp;
				tmp = (*it)->center.y - meanBy[c];
				distB[c] += tmp * tmp;
			}
		}
	}

	double Dist = 0;

	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			Dist += distA[c];
			Dist += distB[c];
		}
	}
	return Dist;
}

double CRTree::distMean(const vector<vector<const PatchFeature *> > &SetA, const vector<vector<const PatchFeature *> > &SetB) {
	// total location entropy (class-independent)
	double meanAx = 0;
	double meanAy = 0;
	int countA = 0;
	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			countA += SetA[c].size();
			for (vector<const PatchFeature *>::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
				meanAx += (*it)->center.x;
				meanAy += (*it)->center.y;
			}
		}
	}

	meanAx /= (double) countA;
	meanAy /= (double) countA;

	double distA = 0;
	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			for (std::vector<const PatchFeature *>::const_iterator it = SetA[c].begin(); it != SetA[c].end(); ++it) {
				double tmp = (*it)->center.x - meanAx;
				distA += tmp * tmp;
				tmp = (*it)->center.y - meanAy;
				distA += tmp * tmp;
			}
		}
	}

	double meanBx = 0;
	double meanBy = 0;
	int countB = 0;
	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			countB += SetB[c].size();
			for (vector<const PatchFeature *>::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
				meanBx += (*it)->center.x;
				meanBy += (*it)->center.y;
			}
		}
	}

	meanBx /= (double) countB;
	meanBy /= (double) countB;

	double distB = 0;
	for (unsigned int c = 0; c < num_labels; ++c) {
		if (class_id[c] > 0) {
			for (std::vector<const PatchFeature *>::const_iterator it = SetB[c].begin(); it != SetB[c].end(); ++it) {
				double tmp = (*it)->center.x - meanBx;
				distB += tmp * tmp;
				tmp = (*it)->center.y - meanBy;
				distB += tmp * tmp;
			}
		}
	}
	return distA + distB;
}

// optimization functions for class impurity

double CRTree::InfGain(const vector<vector<const PatchFeature *> > &SetA, const vector<vector<const PatchFeature *> > &SetB, const std::vector<float> &vRatio) {
	// get size of set A
	double sizeA = 0;
	vector<float> countA(SetA.size(), 0);
	int count = 0;
	for (unsigned int i = 0; i < SetA.size(); ++i) {
		sizeA += float(SetA[i].size()) * vRatio[i];
		if (i > 0 && class_id[i] != class_id[i - 1])
			++count;
		countA[count] += float(SetA[i].size()) * vRatio[i];
	}

	double n_entropyA = 0;
	for (int i = 0; i < count + 1; ++i) {
		double p = double(countA[i]) / sizeA;
		if (p > 0)
			n_entropyA += p * log(p);
	}

	// get size of set B
	double sizeB = 0;
	vector<float> countB(SetB.size(), 0);
	count = 0;
	for (unsigned int i = 0; i < SetB.size(); ++i) {
		sizeB += float(SetB[i].size()) * vRatio[i];
		if (i > 0 && class_id[i] != class_id[i - 1])
			++count;
		countB[count] += float(SetB[i].size()) * vRatio[i];
	}

	double n_entropyB = 0;
	for (int i = 0; i < count + 1; ++i) {
		double p = double(countB[i]) / sizeB;
		if (p > 0)
			n_entropyB += p * log(p);
	}

	return (sizeA * n_entropyA + sizeB * n_entropyB);
}

double CRTree::InfGainBG(const vector<vector<const PatchFeature *> > &SetA, const vector<vector<const PatchFeature *> > &SetB, const std::vector<float> &vRatio) {
	// get size of set A

	double sizeA = 0;
	vector<float> countA(SetA.size(), 0);
	int count = 0;
	for (unsigned int i = 0; i < SetA.size(); ++i) {
		if (i > 0 && ((class_id[i] <= 0 && class_id[i - 1] > 0) || (class_id[i] > 0 && class_id[i - 1] <= 0)))
			++count;

		sizeA += float(SetA[i].size()) * vRatio[i];
		countA[count] += float(SetA[i].size()) * vRatio[i];
	}

	double n_entropyA = 0;
	for (int i = 0; i < count + 1; ++i) {
		double p = double(countA[i]) / sizeA;
		if (p > 0)
			n_entropyA += p * log(p);
	}

	// get size of set B

	double sizeB = 0;
	vector<float> countB(SetB.size(), 0);
	count = 0;
	for (unsigned int i = 0; i < SetB.size(); ++i) {
		if (i > 0 && ((class_id[i] <= 0 && class_id[i - 1] > 0) || (class_id[i] > 0 && class_id[i - 1] <= 0)))
			++count;

		sizeB += float(SetB[i].size()) * vRatio[i];
		countB[count] += float(SetB[i].size()) * vRatio[i];

	}

	double n_entropyB = 0;
	for (int i = 0; i < count + 1; ++i) {
		double p = double(countB[i]) / sizeB;
		if (p > 0)
			n_entropyB += p * log(p);
	}

	return (sizeA * n_entropyA + sizeB * n_entropyB);
}
