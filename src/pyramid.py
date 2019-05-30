import cv2
from skimage import transform

class Pyramid:
	def get_pyramids(self, image, downscale=2):
		pyramid = []
		for (i, resized) in enumerate(transform.pyramid_gaussian(image, downscale)):
			pyramid.append(resized)
			# if the image is too small, break from the loop
			if resized.shape[0] < 2 or resized.shape[1] < 4:
				break
		return pyramid