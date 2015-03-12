#pragma once

#include "CRTree.hpp"

#include <boost/progress.hpp>

#include <memory>
#include <vector>


using namespace std;


class CRForest {
public:
	typedef shared_ptr<CRForest> Ptr;
	typedef shared_ptr<CRForest const> ConstPtr;


	CRForest(int num_trees = 0);
	~CRForest();
	void GetClassID(vector<vector<int> > &id) const;
	unsigned int GetDepth() const;
	bool GetHierarchy(vector<HNode> &hierarchy) const;
	unsigned int GetNumLabels() const;
	int GetSize() const;

	// Regression
	void regression(vector<const LeafNode *> &result, vector<Mat> &vImg, int x, int y) const;
	// Training
	void trainForest(int min_s, int max_d, CvRNG *pRNG, const CRPatch &TrData, int samples, vector<int> &id);

	// IO functions
	void saveForest(const char *filename, unsigned int offset = 0);
	bool loadForest(const char *filename, unsigned int offset = 0);
	void loadHierarchy(const char *hierarchy, unsigned int offset = 0);

	// getter / setter
	const vector<CRTree::Ptr>& getTrees() const;

private:
	/* Trees */
	vector<CRTree::Ptr> vTrees_;

};
