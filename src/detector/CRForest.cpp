#include "CRForest.hpp"

#include <tbb/task_group.h>

CRForest::CRForest(int num_trees) : vTrees_(num_trees) {
}


CRForest::~CRForest() {
	vTrees_.clear();
}


void CRForest::GetClassID(vector<vector<int> > &id) const {
	id.resize(vTrees_.size());
	for (unsigned int i = 0; i < vTrees_.size(); ++i) {
		vTrees_[i]->getClassId(id[i]);
	}
}


unsigned int CRForest::GetDepth() const {
	return vTrees_[0]->GetDepth();
}

bool CRForest::GetHierarchy(vector<HNode> &hierarchy) const {
	return vTrees_[0]->GetHierarchy(hierarchy);
}


unsigned int CRForest::GetNumLabels() const {
	return vTrees_[0]->GetNumLabels();
}


int CRForest::GetSize() const {
	return vTrees_.size();
}


void CRForest::regression(vector<const LeafNode *> &result, vector<Mat> &vImg, int x, int y) const {
	result.resize( vTrees_.size() );
	for (unsigned int i = 0; i < vTrees_.size(); ++i) {
		result[i] = vTrees_[i]->regression(vImg, x, y);
	}
}


void CRForest::trainForest(int min_s, int max_d, CvRNG *pRNG, const CRPatch &TrData, int samples, vector<int> &id) {
	cout << "start training ..." << endl;
	boost::progress_display show_progress( vTrees_.size() );
	tbb::task_group tbb_tg;

	for (int i = 0; i < (int)vTrees_.size(); ++i) {
		function<void()> job_func = [ &, i]() {
			vTrees_[i] = CRTree::Ptr( new CRTree(min_s, max_d, TrData.vLPatches.size(), pRNG));
			vTrees_[i]->setClassId(id);
			vTrees_[i]->growTree(TrData, samples);
			++show_progress;
		};
		tbb_tg.run(bind(job_func));
	}
	tbb_tg.wait();
}


void CRForest::saveForest(const char *filename, unsigned int offset) {
	char buffer[200];
	for (unsigned int i = 0; i < vTrees_.size(); ++i) {
		sprintf_s(buffer, "%s%03d.txt", filename, i + offset);
		vTrees_[i]->saveTree(buffer);
	}
}


bool CRForest::loadForest(const char *filename, unsigned int offset) {
	char buffer[200];
	int cccc = 0;
	bool success = true;
	for (unsigned int i = offset; i < vTrees_.size() + offset; ++i, ++cccc) {
		sprintf_s(buffer, "%s%03d.txt", filename, i);
		bool s;
		vTrees_[cccc] = CRTree::Ptr(new CRTree(buffer, s));
		success *= s;
	}
	return success;
}


void CRForest::loadHierarchy(const char *hierarchy, unsigned int offset) {
	char buffer[400];
	int cccc = 0;
	for (unsigned int i = offset; i < vTrees_.size() + offset; ++i, ++cccc) {
		if (!(vTrees_[cccc]->loadHierarchy(hierarchy))) {
			cerr << "failed to load the hierarchy: " << hierarchy << endl;
		} else {
			cout << "loaded the hierarchy: " << hierarchy << endl;
		}
	}
}


const vector<CRTree::Ptr>& CRForest::getTrees() const {
	return this->vTrees_;
}
