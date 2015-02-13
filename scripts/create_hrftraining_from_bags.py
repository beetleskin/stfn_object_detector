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
		'barrel': "spacebot_barrel.bag",
		'block': "spacebot_block.bag",
	}
	background_bags = {
		'background': "spacebot_background.bag"
	}
	test_bags = {
		'test': "spacebot_testdata.bag"
	}
	

	#
	## identifier for the [rgb, mask and depth] image topics within the bag files, edit if they dont match
	object_topics = ["/camera/rgb/image_color", "/camera/mask", "/camera/depth/image"]
	default_topics = ["/camera/rgb/image_raw", "/camera/depth_registered/image_raw"]


	#
	# other parameter
	max_images_per_object = 200
	max_images_per_background = 100
	max_images_per_test = 100
	output_base_path = "dhrf_config"
	output_data_path = output_base_path + os.sep + "data"
	output_config_path = output_base_path + os.sep + "config"
	
	#####################################################################
	#####################################################################





	# init logging
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

	# prepare the data containers
	object_data = {}
	background_data = {}
	test_data = {}


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
	## read the object bags and extract the images
	logging.info("Reading background data bags")
	for background_id, bag_file in background_bags.items():
		logging.info("\t... reading %s" % bag_file)
		images = load_rgb_depth_from_bag(bag_file, default_topics)
		logging.info("\t... found %d images" % len(images))

		# remove images if too much:
		if len(images) > max_images_per_background:
			logging.info("\t... truncating %d images" % (len(images) - max_images_per_background))
			images = truncate_uniform(images, max_images_per_background)
			images = truncate_uniform(images, max_images_per_background)

		background_data[background_id] = images


	#
	## read the test bags and extract the images
	logging.info("Reading test data bags")
	for test_id, bag_file in test_bags.items():
		logging.info("\t... reading %s" % bag_file)
		images = load_rgb_depth_from_bag(bag_file, default_topics)
		logging.info("\t... found %d images" % len(images))

		# remove images if too much:
		if len(images) > max_images_per_test:
			logging.info("\t... truncating %d images" % (len(images) - max_images_per_test))
			images = truncate_uniform(images, max_images_per_test)
			images = truncate_uniform(images, max_images_per_test)

		test_data[test_id] = images



	#
	## write images and config files
	logging.info("Writing images")


	#
	## check if output path exists and react with override or exit
	if os.path.isdir(output_base_path):
		choices = ['y', 'n']
		user_input = ""
		while user_input not in choices:
			user_input = raw_input("Output directory exists, override? [%s]: " % ('/'.join(choices)))
		if user_input is choices[0]:
			import shutil
			shutil.rmtree(output_base_path, True)
		elif user_input is choices[1]:
			print "Can't write output data, aborting ..."
			sys.exit(0)
	os.mkdir(output_base_path)
	os.mkdir(output_data_path)
	os.mkdir(output_config_path)


	#
	## generate train and test wrapper config
	train_config_file = open(output_config_path + os.sep + "_train_config.cfg",'w')
	train_config_file.write("%d\n" % (len(object_data) + len(background_data)))
	test_config_file = open(output_config_path + os.sep + "_test_config.cfg",'w')
	test_config_file.write("%d\n" % len(test_data))
		
	
	#
	## write object images
	for object_numerical_id, (object_id, object_patches) in enumerate(object_data.items(), 1):
		
		# create sub-directory
		object_data_path = output_data_path + os.sep + object_id
		os.mkdir(object_data_path)

		# init config file
		config_file = open(output_config_path + os.sep + object_id + ".cfg",'w')
		config_file.write("%d 1\n" % len(object_patches))

		# write image and config line
		for i, (rgb, depth) in enumerate(object_patches):
			image_file_base_name = object_id + "_%04i" % i
			config_line = os.path.abspath(object_data_path) + os.sep + image_file_base_name + "_crop.png" + " %d %d %d %d %d %d\n" % (0, 0, rgb.shape[1], rgb.shape[0], int(rgb.shape[1]/2), int(rgb.shape[0]/2))
						
			# write config line
			config_file.write(config_line) 

			# write rgb and depth image
			cv2.imwrite(object_data_path + os.sep + image_file_base_name + "_crop.png", rgb)
			cv2.imwrite(object_data_path + os.sep + image_file_base_name + "_depthcrop.png", depth)

		# add object config to config wrapper file
		train_config_file.write("%d %s\n" % (1, "." + os.sep + os.path.basename(output_config_path) + os.sep + os.path.basename(config_file.name)))



	#
	## write background images
	for background_id, background_images in background_data.items():
		
		# create sub-directory
		background_data_path = output_data_path + os.sep + background_id
		os.mkdir(background_data_path)

		# init config file
		config_file = open(output_config_path + os.sep + background_id + ".cfg",'w')
		config_file.write("%d 1\n" % len(background_images))

		# write image and config line
		for i, (rgb, depth) in enumerate(background_images):
			image_file_base_name = background_id + "_%04i" % i
			config_line = os.path.abspath(background_data_path) + os.sep + image_file_base_name + ".png" + " %d %d %d %d %d %d\n" % (0, 0, rgb.shape[1], rgb.shape[0], int(rgb.shape[1]/2), int(rgb.shape[0]/2))
			
			# write config line
			config_file.write(config_line)

			# write rgb and depth image
			cv2.imwrite(background_data_path + os.sep + image_file_base_name + ".png", rgb)
			cv2.imwrite(background_data_path + os.sep + image_file_base_name + "_depth.png", depth)

		# add object config to config wrapper file
		train_config_file.write("%d %s\n" % (0, "." + os.sep + os.path.basename(output_config_path) + os.sep + os.path.basename(config_file.name)))



	#
	## write test images
	for test_id, test_images in test_data.items():
		
		# create sub-directory
		test_data_path = output_data_path + os.sep + test_id
		os.mkdir(test_data_path)

		# init config file
		config_file = open(output_config_path + os.sep + test_id + ".cfg",'w')
		config_file.write("%d\n" % len(test_images))

		# write image and config line
		for i, (rgb, depth) in enumerate(test_images):
			image_file_base_name = test_id + "_%04i" % i
			config_line = os.path.abspath(test_data_path) + os.sep + image_file_base_name + "_crop.png\n"

			# write config line
			config_file.write(config_line)

			# write rgb and depth image
			cv2.imwrite(test_data_path + os.sep + image_file_base_name + ".png", rgb)
			cv2.imwrite(test_data_path + os.sep + image_file_base_name + "_depth.png", depth)

		test_config_file.write("%s\n" % ("." + os.sep + os.path.basename(output_config_path) + os.sep + os.path.basename(config_file.name)))