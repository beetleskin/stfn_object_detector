#ifndef MODELDBSTUF
#define MODELDBSTUF

#include <string>
#include <vector>

using namespace std;


struct ModelEntry {
	int id;
	string name;
	float radius;
	float threshold_truepositive;
	float threshold_consideration;
	ModelEntry(int id, string name, float radius, float threshold_truepositive, float threshold_consideration) : 
		id(id), name(name), radius(radius), threshold_truepositive(threshold_truepositive), threshold_consideration(threshold_consideration) {}
};


class ModelDBStub {
public:
	ModelDBStub() {
		models.push_back(ModelEntry(0, "soda_can", 0.1, 0.5, 0.25));
		models.push_back(ModelEntry(1, "tissue_can", 0.15, 0.7, 0.35));
	}

	const ModelEntry & get_model_by_id(int id) const {
		return models[id];
	}

private:
	std::vector<ModelEntry> models;
};

#endif
