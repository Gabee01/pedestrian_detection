import os
import io
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

IMAGE_SIZE = (300, 300)
PLOT_LINES = 4
PLOT_COLS = 3

#Enhancement constants
ALPHA = 150
Y = 95

class FingerprintLib:
	def __init__ (self):
		self._fig, self._aplt = plt.subplots(PLOT_LINES, PLOT_COLS)

	# Code to load the databases
	def load_databases(self):
		databasesPath = os.getcwd() + "/databases/"
		databasesList = ["Lindex101/", "Rindex28/"]
		rindexTypeDir = "Rindex28-type/"

		images = []

		for database in databasesList:
			databaseImages = os.listdir(databasesPath + database)
			for image in databaseImages:
				images.append(databasesPath + database + image)

		return images

	def read_raw_image(self, image_path):
		image = np.empty(IMAGE_SIZE, np.uint8)
		image.data[:] = open(image_path).read()
		return image

	# Implement the fingerprint enhancement
	def enhance(self, image):
		(width, height) = image.shape
		enhanced_image = image[:]

		mean = np.mean(image)
		variance = np.var(image)
		for i in range(0,  width):
			for j in range(0,  height):
				if image[i, j] < 2:
					enhanced_image[i, j] = 255
				else:
					s = math.sqrt(variance)
					if (Y < s):
						enhanced_image[i, j] = ALPHA + Y * ((image[i, j] - mean)/s)
					else:
						enhanced_image[i, j] = ALPHA + Y * ((image[i, j] - mean)/Y)

		return enhanced_image

	# Compute the Orientation Map
	def compute_orientation(self, image, block_size):
		image = cv2.medianBlur(image,5)

		alpha_x, alpha_y = self.compute_alpha(image)

		average_x, average_y = self.compute_average(image, alpha_x, alpha_y, block_size)

		gradient_direction = self.compute_gradient(image, average_x, average_y, block_size)
		gradient, image = self.draw_gradient(image, gradient_direction, block_size)
		return (gradient, image, average_x, average_y)


	def compute_gradient(self, image, average_x, average_y, block_size):
		(width, height) = image.shape
		gradient_direction = [[0 for x in range(width/block_size)] for y in range(height/block_size)]
		for i in range(0, width/block_size):
			for j in range(0, height/block_size):
				gradient_direction[i][j] = np.arctan2(average_y[i][j], average_x[i][j]) * .5 + np.pi/2#self.compute_block_angle(average_x[i][j], average_y[i][j])

		return (gradient_direction)

	def compute_average(self, image, alpha_x, alpha_y, block_size):
		(width, height) = image.shape

		average_x =	[[0 for x in range(width/block_size)] for y in range(height/block_size)]
		average_y = [[0 for x in range(width/block_size)] for y in range(height/block_size)]

		for i in range(0, width/block_size):
			for j in range(0, height/block_size):
				for k in range(i*block_size, (i+1) * block_size):
					for l in range(j*block_size, (j+1) * block_size):
						average_x[i][j] += alpha_x[k][l]
						average_y[i][j] += alpha_y[k][l]
				
				average_x[i][j] = average_x[i][j]/pow(block_size, 2)
				average_y[i][j] = average_y[i][j]/pow(block_size, 2)

		return average_x, average_y

	def compute_alpha(self, image):
		(width, height) = image.shape
		sobel = self.create_blank(width, height)

		gx = cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)
		gy = cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)

		alpha_x = gx ** 2 - gy ** 2
		alpha_y = 2 * gx * gy

		return (alpha_x, alpha_y)

	# Load the Fingeprint type annotation
	# Region of interest detection
	def detect_roi(self, image, block_size):
		(width, height) = image.shape
		mean = [[0 for x in range(width/block_size)] for y in range(height/block_size)]
		std_dev = [[0 for x in range(width/block_size)] for y in range(height/block_size)]
		max_mean = 0
		max_std_dev = 0

		for i in range (0, width/block_size):
			for j in range (0, height/block_size):
				block = []
				block_zero = ((i * block_size), (j * block_size))
				block_end = (block_zero[0] + block_size, block_zero[1] + block_size)

				# block = image[[block_zero[0], block_end[0]], :][:,[block_zero[1], block_end[1]]]
				for k in range (block_zero[0], block_end[0]):
					for l in range (block_zero[1], block_end[1]):
						block.append(image[k][l])

				mean[i][j] = np.mean(block)
				std_dev[i][j] = np.std(block)

				if (mean[i][j] > max_mean):
					max_mean = mean[i][j]
				if (std_dev[i][j] > max_std_dev):
					max_std_dev = std_dev[i][j]

		image_center = (width/2, height/2)
		image_center_distance = image.shape[0] * math.sqrt(2) / 2
		valid_blocks = [[0 for x in range(width/block_size)] for y in range(height/block_size)]
		for i in range(0, width/block_size):
			for j in range(0, height/block_size):
				block_zero = ((i * block_size), (j * block_size))
				block_end = (block_zero[0] + block_size, block_zero[1] + block_size)

				block_center = ((block_zero[0] + block_size/2), (block_zero[1] + block_size/2))
				block_ratio_distance = self.get_ratio(image_center, block_center, image_center_distance)

				if (self.is_valid(block_ratio_distance, mean[i][j], max_mean, std_dev[i][j], max_std_dev)):
					valid_blocks[i][j] = 1
		return valid_blocks

	# v = weight_mean (1-u) + weight_std_dev * o + w2
	# weight_mean = 0.5; weight_std_dev = 0.5; w2 = (ratio of the distance to the center)
	# u and o are normalized to be in [0,1]
	# if v > 0.8, the block "is good"
	def is_valid(self, ratio_distance, mean_block, max_mean, std_dev_block, max_std_dev):
		weight_mean = 0.5
		weight_std_dev = 0.5
		
		mean = mean_block/max_mean
		std_dev = std_dev_block/max_std_dev

		v = weight_mean * (1 - mean) + weight_std_dev * std_dev + ratio_distance * 1
		# print("mean_block/max_mean: {}/{} = {}, std_dev_block/max_std_dev: {}/{} = {}, ratio:{}, v:{}"
			# .format(mean_block, max_mean, mean, std_dev_block, max_std_dev, std_dev, ratio_distance, v))

		if v > 0.8:
			return True

	def get_ratio(self, image_center, block_center, greatest_distance):
		block_distance = math.sqrt(math.pow(block_center[0] - image_center[0], 2) + math.pow(block_center[1] - image_center[1], 2))
		# print('block distance = {}'.format(block_distance))
		return 1 - block_distance/greatest_distance
	# Singular point detection (Poincare index)
	def smooth_direction(self, image, average_ax, average_ay, block_size, valid_blocks):
		smoothed_directions = self.compute_smoothed_directions(average_ax, average_ay, block_size, valid_blocks)
		return (smoothed_directions, self.draw_gradient(image, smoothed_directions, block_size, valid_blocks))

	def compute_smoothed_directions(self, alpha_x, alpha_y, block_size, valid_blocks):
		(width, height) = len(alpha_x), len(alpha_x[1])
		smoothed_blocks = [[0 for x in range(width)] for y in range(height)]
		blocks_offset = 1
		for k in range (0, width):
			for l in range (0, height):
				# print (k,l)
				if (valid_blocks[k][l] == 0):
					smoothed_blocks[k][l] = 0
					continue
				# print (k,l,"valid")
				center_block = (k + blocks_offset, l + blocks_offset)
				a = 0
				b = 0

				for m in range(center_block[0] - blocks_offset, center_block[0] + blocks_offset):
					for n in range (center_block[1] - blocks_offset, center_block[1] + blocks_offset):
						if ((m, n) != (center_block[0], center_block[1])):
							a += alpha_x[m][n] 
							b += alpha_y[m][n]

				a += 2 * alpha_x[center_block[0]][center_block[1]]
				b += 2 * alpha_y[center_block[0]][center_block[1]]
				# print ("[{},{}] - b = {}; a = {}; b/a = {}".format(m, n, b, a, b/a))
				smoothed_blocks[k][l] = np.arctan2(b, a)/2 + np.pi/2
		return smoothed_blocks

	def draw_gradient(self, image, gradient_direction, block_size, roi = []):
		(width, height) = image.shape
		gradient_image = np.empty(image.shape, np.uint8)
		gradient_oposite = np.empty(image.shape, np.uint8)
		image_copy = np.copy(image)
		gradient_image.fill(255)
		line_length = block_size / 2 + 1

		for i in range(0, width/block_size):
			for j in range(0, height/block_size):
				if (roi != [] and roi[i][j] == 0):
					continue
				block_center = (i * block_size + block_size/2, j * block_size+block_size/2)
				# print('graditent[{}][{}] = arctan = {} rad'.format(i, j, gradient_direction[i][j]))
				# (x_zero, y_zero) = (i * block_size, j * block_size + block_size)
				(y_zero, x_zero) = block_center

				x = int(x_zero + line_length * math.cos(gradient_direction[i][j]))
				y = int(y_zero + line_length * math.sin(gradient_direction[i][j]))
				cv2.line(image_copy,(x_zero,y_zero), (x, y), (0,255,0), 2)
				cv2.line(gradient_image,(x_zero,y_zero), (x, y), (0,255,0), 2)
				
				# Draw both directions
				gradient_direction[i][j] = gradient_direction[i][j] + np.pi
				x = int(x_zero + line_length * math.cos(gradient_direction[i][j]))
				y = int(y_zero + line_length * math.sin(gradient_direction[i][j]))
				cv2.line(image_copy,(x_zero,y_zero), (x, y), (0,255,0), 2)
				cv2.line(gradient_image,(x_zero,y_zero), (x, y), (0,255,0), 2)

				gradient_direction[i][j] = gradient_direction[i][j] - np.pi

				# print('O = [{},{}], G = [{},{}], degrees = {}'.format(x_zero, y_zero, x, y, math.degrees(gradient_direction[i][j])))
		return (image_copy, gradient_image)
	# Poincare
	def compute_poincare(self, image, angles, valid_blocks, block_size):
		(width, height) = image.shape

		singularities_image = self.create_blank(width, height)

		colors = {"loop" : (150, 0, 0), "delta" : (0, 150, 0), "whorl": (0, 0, 150)}

		for i in range(1, len(angles) - 1):
			for j in range(1, len(angles[i]) - 1):
				if (valid_blocks[i][j] == 0):
					continue
				circle_size = 10
				singularity = self.poincare_index_at(i, j, angles)
				if singularity != "none":
					cv2.circle(singularities_image,((j+1) * block_size, (i+1) * block_size), circle_size, colors[singularity], -1)

		return singularities_image

	def poincare_index_at(self, i, j, angles):
		tolerance = 2
		cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
		deg_angles = [math.degrees(angles[i - k][j - l]) % 180 for k, l in cells]
		index = 0
		for k in range(0, 8):
			if abs(self.get_angle(deg_angles[k], deg_angles[k + 1])) > 90:
				deg_angles[k + 1] += 180
			index += self.get_angle(deg_angles[k], deg_angles[k + 1])

		if 180 - tolerance <= index and index <= 180 + tolerance:
			return "loop"
		if -180 - tolerance <= index and index <= -180 + tolerance:
			return "delta"
		if 360 - tolerance <= index and index <= 360 + tolerance:
			return "whorl"
		return "none"

	def get_angle(self, left, right):
		signum = lambda x: -1 if x < 0 else 1
		angle = left - right
		if abs(angle) > 180:
			angle = -1 * signum(angle) * (360 - abs(angle))
		return angle

	# Fingerprint Type Classification
	# Thining
	def binarize(self, image):
		(width, height) = image.shape
		binarized_image = np.empty(image.shape, np.uint8)
		bins = range(256)
		histogram, _ = np.histogram(image, bins)

		# print("histogram: {}\nTotal = {}".format(histogram, histogram_total))

		quarter_percentile, half_percientile = self.get_percentiles(histogram)

		# self.add_to_plot(histogram, [0,2])
		# print("p25 = {}; p50 = {}".format(quarter_percentile, half_percientile))

		for i in range(0, width):
			for j in range (0, height):
				if (image[i][j] < quarter_percentile):
					binarized_image[i][j] = 0
				elif (image[i][j] > half_percientile):
					binarized_image[i][j] = 255
				else:
					binarized_image[i][j] = self.compare_mean(image, i, j)

		return binarized_image

	def compare_mean(self, image, i, j):
		(width, height) = image.shape
		cells = []
		block_start = -1
		block_end = 1
		for i in range(block_start, block_end):
			for j in range (block_start, block_end):
				if (i <= width and j<=height):
					cells.append((i,j))	
			block_pixels = [image[i - k][j - l] for k, l in cells]

		if (image[i][j] > np.mean(block_pixels)):
			return 255
		else:
			return 0

	def get_percentiles(self, histogram):
		histogram_total = sum(histogram)

		accumulator = 0
		quarter_percentile = 0
		half_percientile = 0
		for i in range(0, len(histogram)):
			accumulator += histogram[i]
			if (quarter_percentile == 0 and accumulator >= histogram_total * .25):
				quarter_percentile = i
			if (half_percientile == 0 and accumulator >= histogram_total * .5):
				half_percientile = i
				return quarter_percentile, half_percientile

	def smooth_binarized_image(self, binarized_image):
		# cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
		smoothed_image = binarized_image[:]
		(width, height) = binarized_image.shape
		for i in range(width):
			for j in range(height):
				white_count, black_count = self.measure_noise(binarized_image, -2, 2)
				if (white_count >= 18):
					smoothed_image[i][j] = 255
				elif (black_count >= 18):
					smoothed_image[i][j] = 255

		very_smoothed_image = smoothed_image[:]
		for i in range(width):
			for j in range(height):
				white_count, black_count = self.measure_noise(smoothed_image, -1, 1)
				if (white_count >= 5):
					very_smoothed_image[i][j] = 255
				elif (black_count >= 5):
					very_smoothed_image[i][j] = 255

		return very_smoothed_image


	def measure_noise(self, binarized_image, block_start, block_end):
		cells = []
		for i in range(block_start, block_end):
			for j in range (block_start, block_end):
				if (i <= width and j<=height):
					cells.append((i,j))

		block_pixels = [binarized_image[i - k][j - l] for k, l in cells]

		black_count = 0
		white_count = 0
		for pixel in block_pixels:
			if (pixel == 255):
				white_count += 1
			if (pixel == 0):
				black_count += 1

		return white_count, black_count

	def skeletonize(self, img):
		size = np.size(img)
		skel = np.zeros(img.shape,np.uint8)
		# ret,img = cv2.threshold(img,127,255,0)
		element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
		done = False
		
		while( not done):
			eroded = cv2.erode(img,element)
			temp = cv2.dilate(eroded,element)
			temp = cv2.subtract(img,temp)
			skel = cv2.bitwise_or(skel,temp)
			img = eroded.copy()

			zeros = size - cv2.countNonZero(img)
			if zeros==size:
				done = True
		
		return skel

	# Minutiae Extraction
	# Pattern Matching

	#General helpers
	def plot(self):
		plt.tight_layout()
		plt.pause(15)
		plt.close()
		self._fig, self._aplt = plt.subplots(PLOT_LINES, PLOT_COLS)

	def add_to_plot(self, image, positionToPlot, title):
		plot = self._aplt[positionToPlot[0], positionToPlot[1]]
		# plot.set_title(title)
		plot.imshow(image, cmap='Greys_r')

	def create_blank(self, width, height):
		blank_image = np.zeros((height, width, 3), np.uint8)
		return blank_image