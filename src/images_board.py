import matplotlib.pyplot as plt
import cv2

class ImagesBoard:
    def __init__ (self):
        PLOT_LINES = 4
        PLOT_COLS = 3
        cv2.destroyAllWindows()
        self._fig, self._aplt = plt.subplots(PLOT_LINES, PLOT_COLS)

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

    def plot_image_and_hist(self, image, histogram, bins):
        # print(histogram_bins)
        # plt.hist(histogram_bins, bins_angles, weights = bins_angles)
        fig = plt.figure()
        fig = plt.bar(bins, histogram)
        # resized_image = cv2.resize(image, (640, 1280))
        cv2.imshow('img', image)
        # # cv2.waitKey(0)
        # cv2.destroyAllWindows()