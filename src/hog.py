import cv2
import numpy as np
import operator
from images_board import *
import math

class Hog:
	def __init__(self):
		self.board = ImagesBoard()
		self.bins_angles = [math.radians(angle) for angle in [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]]
		# self.bins_angles = [10, 30, 50, 70, 90, 110, 130, 150, 170]

	def get_block_bins(self, image, window_size):
		histogram_bins = [0] * 9
		gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
		gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)

		mag, angle = cv2.cartToPolar(gx, gy)

		for i in range(window_size):
			for j in range(window_size):

				left_bin = 0
				right_bin = 0
				chosen_angle = 0
				chosen_mag = 0
				mag_index, chosen_mag = max(enumerate(mag[i][j]), key=operator.itemgetter(1))
				
				chosen_angle = angle[i][j][mag_index]
				if (chosen_angle > math.radians(180)):
					chosen_angle = chosen_angle - math.radians(180)

				for bin_index in range(len(self.bins_angles)):

					if (self.bins_angles[bin_index] < chosen_angle):
						continue

					if (chosen_angle == 0):
						right_bin = 0
						left_bin = 0
						break

					else:
						right_bin = bin_index
						left_bin = bin_index - 1

					break

				left_dif = chosen_angle - self.bins_angles[left_bin]
				right_dif = self.bins_angles[right_bin] - chosen_angle
				max_dif = self.bins_angles[right_bin] - self.bins_angles[left_bin]

				print(left_dif, right_dif)
				if (right_bin == len(self.bins_angles) - 1):
						right_bin = 0

				if (left_bin == len(self.bins_angles) - 1):
						left_bin = 0

				if (left_bin == right_bin):
					histogram_bins[left_bin] += abs(chosen_mag)
				else:
					histogram_bins[left_bin] += abs(chosen_mag * (left_dif/max_dif))
					histogram_bins[right_bin] += abs(chosen_mag * (right_dif/max_dif))
		return histogram_bins

	def draw_hog(self, image, i, j, histogram, block_size):
		block_center = (i + block_size/2, j + block_size/2)
		width, height, _ = image.shape
		gradient_image = image[:]
		line_length = math.ceil(block_size / 2)

		for k in range(0, len(histogram)):
			magnitude = histogram[k]
			angle = self.bins_angles[k]
			(y_zero, x_zero) = block_center

			angle += np.pi/2
			x = int(x_zero + line_length * magnitude * math.cos(angle))
			y = int(y_zero + line_length * magnitude * math.sin(angle))

			cv2.line(gradient_image,(x_zero,y_zero), (x, y), (magnitude,magnitude,magnitude))

			angle += np.pi

			x = int(x_zero + line_length * magnitude * math.cos(angle))
			y = int(y_zero + line_length * magnitude * math.sin(angle))
			cv2.line(gradient_image,(x_zero,y_zero), (x, y), (magnitude,magnitude,magnitude))

		# cropped_image = gradient_image[i:i+block_size, j:j+block_size]
		# self.board.plot_image_and_hist(cropped_image, histogram, self.bins_angles[:-1])

		return gradient_image


	def compute(self, image):
		# print(histogram_bins)
		image = np.float32(image)# / 255
		height, width, _ = image.shape
		block_size = 8
		amplify_times = 8
		resized_image = cv2.resize(image, (image.shape[1] * amplify_times, image.shape[0] * amplify_times))

		blocks_histograms = []
		max_magnitude = 0
		for i in range(0, height, block_size): # 128/8 = 16
			for j in range(0, width, block_size): # 64/8 = 8
				window = image[i:i+block_size, j:j+block_size]
				block_histogram = self.get_block_bins(window, block_size)
				max_magnitude = max(max(block_histogram), max_magnitude)
				blocks_histograms.append(block_histogram)

		current_block = 0
		for i in range(0, height, block_size): # 128/8 = 16
			for j in range(0, width, block_size): # 64/8 = 8
				block_histogram = blocks_histograms[current_block]
				current_block += 1
				block_histogram = [magnitude/max_magnitude for magnitude in block_histogram]
				image = self.draw_hog(resized_image, i * amplify_times, j * amplify_times, block_histogram, block_size * amplify_times)


		cv2.imshow('img' + str(image.shape), resized_image)
		cv2.waitKey(0)
		return (blocks_histograms, self.bins_angles[:-1])

		# return (mag, angle)
		# g = sqrt(pow(gx,2) + pow(gy,2))
		# angle = math.arctan2(gy/gx)
