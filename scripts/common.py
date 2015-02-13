#!/usr/bin/env python
import cv2
import numpy as np

def scale_rect(rect, scaleX, scaleY=None):
	if scaleY is None: scaleY = scaleX
	dx = rect[2]*scaleX - rect[2]
	dy = rect[3]*scaleY - rect[3]
	rect[0] -= int(np.round(dx*0.5))
	rect[1] -= int(np.round(dy*0.5))
	rect[2] += int(np.round(dx))
	rect[3] += int(np.round(dy))


def move_rect(rect, d):
	rect[0] += d[0]
	rect[1] += d[1]