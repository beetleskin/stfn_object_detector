#ifndef MODELDBSTUF
#define MODELDBSTUF

#include <string>
#include <vector>

using namespace std;


struct ModelEntry {
	const int id;
	const string name;
	const float radius;
	const float threshold_truepositive;
	const float threshold_consideration;
	const string pcd_model_file;
	ModelEntry(int id, string name, float radius, float threshold_truepositive, float threshold_consideration, const string pcd_model_file) : 
		id(id), name(name), radius(radius), threshold_truepositive(threshold_truepositive), threshold_consideration(threshold_consideration), pcd_model_file(pcd_model_file) {}
};


class ModelDBStub {
public:
	ModelDBStub() {
		models.push_back(ModelEntry(0, "soda_can", 0.1, 0.5, 0.25, "/home/stfn/testdata/can_1_cloud.pcd"));
		models.push_back(ModelEntry(1, "tissue_can", 0.15, 0.7, 0.35, ""));
	}

	const ModelEntry & get_model_by_id(int id) const {
		return models[id];
	}

private:
	std::vector<ModelEntry> models;
};

#endif
