#include <iostream>
#include <fstream>

using namespace std;

class TrainStats {
public:
	static TrainStats &get() {
		static TrainStats instance;
		return instance;
	}

	void addSplit(int treeNo, int nodeID, int measureMode);
	void addMeasure(int treeNo, int channel, int measureMode, float distance);
	void addInvalidTest(int treeNo, int nodeID);

private:
	TrainStats();
	~TrainStats();
	TrainStats(TrainStats const &);
	void operator=(TrainStats const &);

	/*member*/
	vector<string> data;
	vector<ofstream *> splitOutFiles;
	vector<ofstream *> measureOutFiles;
	vector<ofstream *> invOutFiles;
};





TrainStats::TrainStats() {
	for (int i = 0; i < 15; ++i) {
		stringstream splitSs, measureSs, invSs;
		splitSs << "train_stats_raw/trainstats_split_" << i << ".txt";
		measureSs << "train_stats_raw/trainstats_measure_" << i << ".txt";
		invSs << "train_stats_raw/invalid_tests_" << i << ".txt";

		splitOutFiles.push_back(new ofstream(splitSs.str().c_str()));
		measureOutFiles.push_back(new ofstream(measureSs.str().c_str()));
		invOutFiles.push_back(new ofstream(invSs.str().c_str()));
	}

}

TrainStats::~TrainStats() {
	for (int i = 0; i < splitOutFiles.size(); ++i) {
		splitOutFiles[i]->close();
		delete splitOutFiles[i];
		measureOutFiles[i]->close();
		delete measureOutFiles[i];
	}

	splitOutFiles.clear();
	measureOutFiles.clear();


}


void TrainStats::addSplit(int treeNo, int nodeID, int measureMode) {
	*splitOutFiles[treeNo] << nodeID << " " << measureMode << "\n";
}


void TrainStats::addMeasure(int treeNo, int channel, int measureMode, float distance) {
	*measureOutFiles[treeNo] << channel << " " << measureMode << " " << distance << "\n";
}

void TrainStats::addInvalidTest(int treeNo, int nodeID) {
	*invOutFiles[treeNo] << nodeID << "\n";
}

