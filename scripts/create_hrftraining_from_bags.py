#!/usr/bin/env python
import rospy
import rosbag

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import logging
import cv2
import numpy as np
import os
import sys

import common

"""


"""


def prepare_path(path, model):
	""""
	Checks if path exists and react with override or return, dependent on user choice.
	"""

	if os.path.isdir(path):
		choices = ['y', 'n']
		user_input = ""
		while user_input not in choices:
			user_input = raw_input("DB folder for model '%s' exists, override? [%s]: " % (model, '/'.join(choices)))
		if user_input is choices[0]:
			import shutil
			shutil.rmtree(path, True)
		elif user_input is choices[1]:
			return False

	os.mkdir(path)
	return True
	





def truncate_uniform(list, max_elements):
	"""
	Truncates a list to the size of max_elements if the list is longer than max_elements.
	The items that are to be removed are uniformly distributed over the list. 
	When list has 300 items, and max_elements is 200, every 3rd item is removed.
	"""

	list_length = len(list)

	if list_length > max_elements:
		to_remove_idx = np.linspace(0, list_length, list_length-max_elements, endpoint=False)
		to_remove_idx = to_remove_idx.astype(np.int32)
		list = [x for (i, x) in enumerate(list) if i not in to_remove_idx]

	return list



def load_rgb_depth_from_bag(bagFile, topics):
	"""
	Extracts image messages from the given topics from the bag file.
	It is assumed, that topics has two elements:
		topics[0] => the topic of the rgb image (bgr8)
		topics[1] => the topic of the depth image (uint16 or float32)
	"""
	bag = rosbag.Bag(bagFile, "r")
	bridge = CvBridge()

	dataRGB = []
	dataDepth = []


	for _, msg, _ in bag.read_messages(topics=[topics[0]]):
		cv_mat = bridge.imgmsg_to_cv2(msg, "bgr8")
		np_image = np.asarray(cv_mat, np.uint8)
		dataRGB.append(np_image)
	for _, msg, _ in bag.read_messages(topics=[topics[1]]):
		if (msg.step / msg.width * 8 is 16):
			np_image = np.ndarray(shape=(msg.height, msg.width), dtype=np.uint16, order="C", buffer=msg.data)
		else:
			np_image = np.ndarray(shape=(msg.height, msg.width), dtype=np.float32, order="C", buffer=msg.data)
			np_image = np.nan_to_num(np_image)
			np_image = (np_image * 1000).astype(np.uint16)

		dataDepth.append(np_image)

	bag.close()
	return zip(dataRGB, dataDepth)



def load_rgb_mask_depth_from_bag(bagFile, topics):
	"""
	Extracts image messages from the given topics from the bag file.
	It is assumed, that topics has three elements:
		topics[0] => the topic of the rgb image (bgr8)
		topics[0] => the topic of the mask image (mono8)
		topics[1] => the topic of the depth image (uint16 or float32)
	"""
	bag = rosbag.Bag(bagFile, "r")
	bridge = CvBridge()

	dataRGB = []
	dataMask = []
	dataDepth = []
	
	for _, msg, _ in bag.read_messages(topics=[topics[0]]):
		cv_mat = bridge.imgmsg_to_cv2(msg, "bgr8")
		np_image = np.asarray(cv_mat, np.uint8)
		dataRGB.append(np_image)
	for _, msg, _ in bag.read_messages(topics=[topics[1]]):
		cv_mat = bridge.imgmsg_to_cv2(msg, "mono8")
		np_image = np.asarray(cv_mat, np.uint8)
		dataMask.append(np_image)
	for _, msg, _ in bag.read_messages(topics=[topics[2]]):
		if (msg.step / msg.width * 8 is 16):
			np_image = np.ndarray(shape=(msg.height, msg.width), dtype=np.uint16, order="C", buffer=msg.data)
		else:
			np_image = np.ndarray(shape=(msg.height, msg.width), dtype=np.float32, order="C", buffer=msg.data)
			np_image = np.nan_to_num(np_image)
			np_image = (np_image * 1000).astype(np.uint16)
		dataDepth.append(np_image)

	bag.close()
	return zip(dataRGB, dataMask, dataDepth)



def extract_positive_images_from_labeled_data(data):
	"""
	"""

	#depth_cap = 20000.0
	#normalized_patch_size = (64, 128)
	roi_scale = 1.3
	patches = []


	for i, (frame_rgb, frame_mask, frame_depth) in enumerate(data):
		if np.sum(frame_mask) == 0:
			logging.warning("skipping empty frame")
			continue
		#frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
		#frame_depth_normalized = (frame_depth.astype(np.float32) / depth_cap * 255).astype(np.uint8)
		#cv2.imshow("frame masked before", frame_gray * (frame_mask/255))
		#cv2.imshow("frame", frame_gray)

		# 1: smooth and erode the mask
		frame_mask = cv2.GaussianBlur(frame_mask, (5,5), 1.5)
		frame_mask = cv2.GaussianBlur(frame_mask, (3,3), 1.2)
		_,frame_mask = cv2.threshold(frame_mask, 32, 255., cv2.THRESH_BINARY)
		frame_mask = cv2.dilate(frame_mask, np.ones((3,3)))
		frame_mask = cv2.erode(frame_mask, np.ones((5,5)))
		frame_mask = cv2.erode(frame_mask, np.ones((5,5)))

		#cv2.imshow("frame masked after", frame_gray * (frame_mask/255))
		#cv2.waitKey(10000)

		# find bounding rect of object
		contours, _ = cv2.findContours( frame_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		roi = list(cv2.boundingRect(contours[0]))
		try:
			common.scale_rect(roi, roi_scale, roi_scale)
			patch_rgb = frame_rgb[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2],:]
			patch_depth = frame_depth[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
			patches.append([patch_rgb, patch_depth])
		except Exception, e:
			logging.warning(e)
			continue

	return patches

	

def unify_patch_size(patches):

	# find averages and std deviation
	av_shape = np.array([0,0])
	av_depth = 0
	std_dev_shape = np.array([0,0])
	for (rgb, depth) in patches:
		av_shape += [rgb.shape[0], rgb.shape[1]]
		av_depth += np.mean(depth[depth.shape[0]/2-1:depth.shape[0]/2+2,depth.shape[1]/2-1:depth.shape[1]/2+2])
	av_shape /= len(patches)
	av_depth /= len(patches)

	for (rgb, depth) in patches:
		std_dev_shape += np.abs(av_shape - np.array([rgb.shape[0], rgb.shape[1]]))
	std_dev_shape /= len(patches)

	# remove odd patches
	to_remove = []
	for i, (rgb, depth) in enumerate(patches):
		diff = np.abs(av_shape - np.array([rgb.shape[0], rgb.shape[1]]))
		if diff[0] > std_dev_shape[0] or diff[1] > std_dev_shape[1]:
			to_remove.append(i)

	for i in sorted(to_remove, reverse=True):
		del patches[i]

	
	print "mean depth: " + str(av_depth)
	print "mean size: " + str(av_shape)
	print "std dev size: " + str(std_dev_shape)

	# scale patches to unit-size
	for i, (rgb, depth) in enumerate(patches):
		scale = float(av_shape[0]) / rgb.shape[0]
		new_shape = (int(rgb.shape[1]*scale), int(rgb.shape[0]*scale))
		patches[i][0] = cv2.resize(rgb, new_shape)
		patches[i][1] = cv2.resize(depth, new_shape)




if __name__ == '__main__':

	
	#####################################################################
	########################### USER EDIT AREA ###########################


	#
	## put the tain and thest data in here
	object_bags = {
		'can': "can.bag",
		'tissue_can': "tissue_can.bag",
	}
	scene_bags = {
		#'spacebot_bg': "spacebot_background.bag",
		#'spacebot_test': "spacebot_testdata.bag"
	}
	

	#
	## identifier for the [rgb, mask and depth] image topics within the bag files, edit if they dont match
	object_topics = ["/camera/rgb/image_color", "/camera/mask", "/camera/depth/image"]
	scene_topics = ["/camera/rgb/image_raw", "/camera/depth_registered/image_raw"]


	#
	# other parameter
	db_basepath = rospy.get_param("/recognition_db_basepath", "/home/stfn/recognition_database")
	max_images_per_object = 200
	max_images_per_scene = 100
	max_images_per_test = 100
	
	#####################################################################
	#####################################################################





	# init logging
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	# prepare the data containers
	object_data = {}
	scene_data = {}


	#
	## read the object bags, extract the images and extract patches with the help of the foreground object mask
	logging.info("Reading object data bags")
	for object_id, bag_file in object_bags.items():
		logging.info("\t... reading %s" % bag_file)
		images = load_rgb_mask_depth_from_bag(bag_file, object_topics)
		logging.info("\t... found %d images" % len(images))

		# remove images if too much:#
		if len(images) > max_images_per_object:
			logging.info("\t... truncating %d images" % (len(images) - max_images_per_object))
			images = truncate_uniform(images, max_images_per_object)
			images = truncate_uniform(images, max_images_per_object)
			images = truncate_uniform(images, max_images_per_object)

		logging.info("\t... extracting object patches")
		patches = extract_positive_images_from_labeled_data(images)
		#unify_patch_size(object_patches)
		object_data[object_id] = patches


	#
	## read the scene bags and extract the images
	logging.info("Reading scene data bags")
	for scene_id, bag_file in scene_bags.items():
		logging.info("\t... reading %s" % bag_file)
		images = load_rgb_depth_from_bag(bag_file, scene_topics)
		logging.info("\t... found %d images" % len(images))

		# remove images if too much:
		if len(images) > max_images_per_scene:
			logging.info("\t... truncating %d images" % (len(images) - max_images_per_scene))
			images = truncate_uniform(images, max_images_per_scene)
			images = truncate_uniform(images, max_images_per_scene)

		scene_data[scene_id] = images




	#
	## write images and config files
	logging.info("Writing images")

	
	#
	## write object images
	for object_id, object_patches in object_data.items():


		# prepare path
		db_model_base_path = db_basepath + os.sep + "models" + os.sep + object_id
		db_model_image_path = db_model_base_path + os.sep + 'images'
		prepared = prepare_path(db_model_base_path, object_id)
		if not prepared:
			logging.warning("\tskipping %s" % object_id)
			continue
		os.mkdir(db_model_image_path)


		# init config file
		config_file = open(db_model_base_path + os.sep + object_id + ".cfg",'w')

		# write image and config line
		for i, (rgb, depth) in enumerate(object_patches):
			image_file_base = db_model_image_path + os.sep + object_id + "_%04i" % i
			config_line = image_file_base + "_crop.png" + " %d %d %d %d %d %d\n" % (0, 0, rgb.shape[1], rgb.shape[0], int(rgb.shape[1]/2), int(rgb.shape[0]/2))
						
			# write config line
			config_file.write(config_line) 

			# write rgb and depth image
			cv2.imwrite(image_file_base + "_crop.png", rgb)
			cv2.imwrite(image_file_base + "_depthcrop.png", depth)



	#
	## write background images
	for scene_id, scene_images in scene_data.items():

		# prepare path
		db_model_base_path = db_basepath + os.sep + "scenes" + os.sep + scene_id
		db_model_image_path = db_model_base_path + os.sep + 'images'
		prepared = prepare_path(db_model_base_path, scene_id)
		if not prepared:
			logging.warning("\tskipping %s" % scene_id)
			continue
		os.mkdir(db_model_image_path)
		

		# init config file
		config_file = open(db_model_base_path + os.sep + scene_id + ".cfg",'w')

		# write image and config line
		for i, (rgb, depth) in enumerate(scene_images):
			image_file_base = os.path.abspath(db_model_image_path) + os.sep + scene_id + "_%04i" % i
			config_line = image_file_base + ".png\n"

			# write config line
			config_file.write(config_line)

			# write rgb and depth image
			cv2.imwrite(image_file_base + ".png", rgb)
			cv2.imwrite(image_file_base + "_depth.png", depth)