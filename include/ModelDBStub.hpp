#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ros/ros.h>



struct ModelEntry {
	std::string class_name;
	float radius;
	float threshold_truepositive;
	float threshold_consideration;
	std::string pcd_model_file;
	ModelEntry(){};
	ModelEntry(std::string class_name, float radius, float threshold_truepositive, float threshold_consideration, const std::string pcd_model_file) : 
		class_name(class_name), radius(radius), threshold_truepositive(threshold_truepositive), threshold_consideration(threshold_consideration), pcd_model_file(pcd_model_file) {}
};


class ModelDBStub {
public:
	typedef std::shared_ptr<ModelDBStub> Ptr;
	typedef std::shared_ptr<ModelDBStub const> ConstPtr;

	static ModelDBStub::Ptr get() {
		static ModelDBStub::Ptr instance(new ModelDBStub);
		return instance;
	}

	const ModelEntry & get_model(const std::string & class_name) {
		return models[class_name];
	}

protected:
	ModelDBStub() {
		std::vector<std::string> model_class_names;
		ros::param::get("~models", model_class_names);


		for (int i = 0; i < model_class_names.size(); ++i) {
			std::string class_name = model_class_names[i];
			int radius = -1;
			float threshold_truepositive = -1;
			float threshold_consideration = -1;
			std::string pcd_model_file = "";

			models[class_name] = ModelEntry( class_name, radius, threshold_truepositive, threshold_consideration, pcd_model_file);
		}
	}


private:
	std::map<std::string, ModelEntry> models;
};