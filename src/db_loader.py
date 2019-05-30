from matplotlib import pyplot as plt
import os
import io
import numpy as np

# Code to load the databases
class DbLoader:
    def __init__(self):
        self.annotations_path = os.getcwd() + '/../databases/INRIAPerson/Train/annotations/'
        self.databasesPath = os.getcwd() + "/../databases/INRIAPerson/"
        self.databasesList = ["70X134H96/Test/pos/", "96X160H96/Train/pos/"]

    def load_databases(self):
        images = []
        # print(self.databasesPath)
        for database in self.databasesList:
            databaseImages = os.listdir(self.databasesPath + database)
            # print(databaseImages)
            for image in databaseImages:
                images.append(self.databasesPath + database + image)

        return images

    def read_raw_image(self, image_annotation):
        image = np.empty(image_annotation.image_size, np.uint8)
        image.data[:] =self.open(image_annotation.file_name).read()
        return image

    def load_annotations(self):
        annotations = []

        annotations_list = os.listdir(self.annotations_path)
        # print(annotations_list)
        for annotation in annotations_list:
            annotations.append(self.annotations_path + annotation)

        return annotations


    def read_file(self, path):
        print("Reading {} ...".format(path))
        with io.open(path, 'r', encoding="ISO-8859-1") as file:
            file_data = file.read()
        return file_data