#pragma once
#define sprintf_s sprintf

#include "CRPatch.hpp"

#include <fstream>
#include <iostream>
#include <memory>


using namespace std;


// Auxilary structure
struct IntIndex {
	int val;
	unsigned int index;
	bool operator<(const IntIndex &a) const {
		return val < a.val;
	}
};

// Structure for the leafs
struct LeafNode {
	// Constructors
	LeafNode() {}

	// IO functions
	const void show(int delay, int width, int height, int *class_id);
	const void print() const {
		cout << "Leaf " << vCenter.size() << " ";
		for (unsigned int c = 0; c < vCenter.size(); ++c)
			cout << vCenter[c].size() << " "  << vPrLabel[c] << " ";
		cout << endl;
	}
	float cL; // what proportion of the entries at this leaf is from foreground
	int idL; // leaf id
	float fL; //occurrence frequency
	float eL;// emprical probability of when a patch is matched to this cluster, it belons to fg
	vector<int> nOcc;
	vector<float> vLabelDistrib;
	// Probability of foreground
	vector<float> vPrLabel;
	// Vectors from object center to training patches
	vector<vector<Point> > vCenter;
	vector<vector<float> > vCenterWeights;
	vector<vector<int> > vID;
};

// Structure for internal Nodes
struct InternalNode {
	// Constructors
	InternalNode() {}

	// Copy Constructor
	InternalNode(const InternalNode &arg) {
		parent = arg.parent;
		leftChild = arg.leftChild;
		rightChild = arg.rightChild;
		idN = arg.idN;
		depth = arg.depth;
		data.resize(arg.data.size());
		for (unsigned int dNr = 0; dNr < arg.data.size(); dNr++)
			data[dNr] = arg.data[dNr];
		isLeaf = arg.isLeaf;
	}

	// relative node Ids
	int parent; // parent id, if this node is root, the parent will be -1
	int leftChild; // stores the left child id, if leaf stores the leaf id
	int rightChild;// strores the right child id, if leaf is set to -1

	//internal data
	int idN;//node id

	int depth;

	//  the data inside each not
	vector<int> data;// x1 y1 x2 y2 channel threshold
	bool isLeaf;// if leaf is set to 1 otherwise to 0, the id of the leaf is stored at the left child

};

struct HNode {
	HNode() {}

	// explicit copy constructor
	HNode(const HNode &arg) {
		id = arg.id;
		parent = arg.parent;
		leftChild = arg.leftChild;
		rightChild = arg.rightChild;
		subclasses = arg.subclasses;
		linkage = arg.linkage;
	}

	bool isLeaf() {
		return ((leftChild < 0) && (rightChild < 0));
	}

	int id;
	int parent;// stores the id of the parent node: if root -1
	int leftChild; // stores the id of the left child, if leaf -1
	int rightChild;// stores the id of the right child, if leaf -1
	float linkage;
	vector<int> subclasses; // stores the id of the subclasses which are under this node,
};


class CRTree {
public:
	typedef shared_ptr<CRTree> Ptr;
	typedef shared_ptr<CRTree const> ConstPtr;

	// Constructors
	CRTree(const char *filename, bool &success);
	CRTree(int min_s, int max_d, int l, CvRNG *pRNG) : min_samples(min_s), max_depth(max_d), num_leaf(0), num_nodes(1), num_labels(l), cvRNG(pRNG) {
		this->id = CRTree::treeCount++;

		nodes.resize(int(num_nodes));
		nodes[0].isLeaf = false;
		nodes[0].idN = 0; // the id is set to zero for the root
		nodes[0].leftChild = -1;
		nodes[0].rightChild = -1;
		nodes[0].parent = -1;
		nodes[0].data.resize(6, 0);
		nodes[0].depth = 0;

		//initializing the leafs
		leafs.resize(0);
		// class structure
		class_id = new int[num_labels];
	}
	~CRTree() {
		delete[] class_id;   //clearLeaves(); clearNodes();
	}

	// Set/Get functions
	unsigned int GetDepth() const {
		return max_depth;
	}
	unsigned int GetNumLabels() const {
		return num_labels;
	}
	void setClassId(vector<int> &id) {
		for (unsigned int i = 0; i < num_labels; ++i) class_id[i] = id[i];
	}
	void getClassId(vector<int> &id) const {
		id.resize(num_labels);
		for (unsigned int i = 0; i < num_labels; ++i) id[i] = class_id[i];
	}
	

	int getNumLeaf() {
		return num_leaf;
	}

	LeafNode *getLeaf(int leaf_id = 0) {
		return &leafs[leaf_id];
	}

	bool GetHierarchy(vector<HNode> &h) {
		if ( (hierarchy.size() == 0) ) { // check if the hierarchy is set at all(hierarchy == NULL) ||
			return false;
		}
		h = hierarchy;
		return true;
	};

	// Regression
	const LeafNode *regression(vector<Mat> &vImg, int x, int y) const;

	// Training
	void growTree(const CRPatch &TrData, int samples);

	// IO functions
	bool saveTree(const char *filename) const;
	bool loadHierarchy(const char *filename);

private:

	static unsigned int treeCount;
	unsigned int id;

	// Private functions for training
	void grow(const vector<vector<const PatchFeature *> > &TrainSet, const vector<vector<int> > &TrainIDs, int node, unsigned int depth, int samples, vector<float> &vRatio);

	int getStatSet(const vector<vector<const PatchFeature *> > &TrainSet, int *stat);

	void makeLeaf(const vector<vector<const PatchFeature *> > &TrainSet, const vector<vector< int> > &TrainIDs , vector<float> &vRatio, int node);

	bool optimizeTest(vector<vector<const PatchFeature *> > &SetA, vector<vector<const PatchFeature *> > &SetB, vector<vector<int> > &idA, vector<vector<int> > &idB, const vector<vector<const PatchFeature *> > &TrainSet, const vector<vector<int> > &TrainIDs, int *test, unsigned int iter, unsigned int mode, const vector<float> &vRatio);



	void generateTest(int *test, unsigned int max_w, unsigned int max_h, unsigned int max_c);

	void evaluateTest(vector<vector<IntIndex> > &valSet, const int *test, const vector<vector<const PatchFeature *> > &TrainSet);

	void split(vector<vector<const PatchFeature *> > &SetA, vector<vector<const PatchFeature *> > &SetB, vector<vector<int > > &idA, vector<vector<int> > &idB , const vector<vector<const PatchFeature *> > &TrainSet, const vector<vector<IntIndex> > &valSet, const vector<vector<int> > &TrainIDs , int t);

	double measureSet(const vector<vector<const PatchFeature *> > &SetA, const vector<vector<const PatchFeature *> > &SetB, unsigned int mode, const vector<float> &vRatio) {
		if (mode == 0) {
			return InfGainBG(SetA, SetB, vRatio) + InfGain(SetA, SetB, vRatio) / double(SetA.size());
		} else {
			return -distMeanMC(SetA, SetB);
		}
	}

	double distMean(const vector<vector<const PatchFeature *> > &SetA, const vector<vector<const PatchFeature *> > &SetB);
	double distMeanMC(const vector<vector<const PatchFeature *> > &SetA, const vector<vector<const PatchFeature *> > &SetB);

	double InfGain(const vector<vector<const PatchFeature *> > &SetA, const vector<vector<const PatchFeature *> > &SetB, const vector<float> &vRatio);

	double InfGainBG(const vector<vector<const PatchFeature *> > &SetA, const vector<vector<const PatchFeature *> > &SetB, const vector<float> &vRatio);

	// Data structure

	// tree table
	// 2^(max_depth+1)-1 x 7 matrix as vector
	// column: leafindex x1 y1 x2 y2 channel thres
	// if node is not a leaf, leaf=-1
	//int* treetable;

	// stop growing when number of patches is less than min_samples
	unsigned int min_samples;

	// depth of the tree: 0-max_depth
	unsigned int max_depth;

	// number of nodes: 2^(max_depth+1)-1
	unsigned int num_nodes;

	// number of leafs
	unsigned int num_leaf;

	// number of labels
	unsigned int num_labels;

	// classes
	int *class_id;

	//leafs as vector
	vector<LeafNode> leafs;

	// internalNodes as vector
	vector<InternalNode> nodes;// the first element of this is the root

	// hierarchy as vector
	vector<HNode> hierarchy;
	CvRNG *cvRNG;

	static const int depth_channel = 60;
};


inline const LeafNode *CRTree::regression(vector<Mat> &vImg, int x, int y) const {
	int node = 0;



	// Go through tree until one arrives at a leaf, i.e. pnode[0]>=0)
	while (!nodes[node].isLeaf) {
		// binary test 0 - left, 1 - right
		// Note that x, y are changed since the patches are given as matrix and not as image
		// p1 - p2 < t -> left is equal to (p1 - p2 >= t) == false

		// depth scale value
		float depth_scale = vImg[depth_channel].ptr<float>(y + 5)[x + 5];
		// get pixel values
		float x1 = nodes[node].data[0];
		float y1 = nodes[node].data[1];
		float x2 = nodes[node].data[2];
		float y2 = nodes[node].data[3];
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
			x1 += 5;
			y1 += 5;
			x2 += 5;
			y2 += 5;
		}

		x1 += x + 0.5;
		y1 += y + 0.5;
		x2 += x + 0.5;
		y2 += y + 0.5;
		if (x1 >= vImg[0].cols)
			x1 = vImg[0].cols - 1;
		if (y1 >= vImg[0].rows)
			y1 = vImg[0].rows - 1;
		if (x2 >= vImg[0].cols)
			x2 = vImg[0].cols - 1;
		if (y2 >= vImg[0].rows)
			y2 = vImg[0].rows - 1;
		if (x1 < 0)
			x1 = 0;
		if (y1 < 0)
			y1 = 0;
		if (x2 < 0)
			x2 = 0;
		if (y2 < 0)
			y2 = 0;

		int p1 = vImg[nodes[node].data[4]].ptr<uchar>(int(y1))[int(x1)];
		int p2 = vImg[nodes[node].data[4]].ptr<uchar>(int(y2))[int(x2)];
		// test
		bool test = ( p1 - p2 ) >= nodes[node].data[5];

		// next node is at the left or the right child depending on test
		if (test)
			node = nodes[node].rightChild;
		else
			node = nodes[node].leftChild;
	}


	return &leafs[nodes[node].leftChild];
}


inline void CRTree::generateTest(int *test, unsigned int max_w, unsigned int max_h, unsigned int max_c) {
	test[0] = cvRandInt( cvRNG ) % max_w;
	test[1] = cvRandInt( cvRNG ) % max_h;
	test[2] = cvRandInt( cvRNG ) % max_w;
	test[3] = cvRandInt( cvRNG ) % max_h;
	test[4] = cvRandInt( cvRNG ) % max_c;
}


inline int CRTree::getStatSet(const vector<vector<const PatchFeature *> > &TrainSet, int *stat) {
	int count = 0;
	for (unsigned int l = 0; l < TrainSet.size(); ++l) {
		if (TrainSet[l].size() > 0)
			stat[count++] = l;
	}
	return count;
}
