import re
from image import *

class Parser:

    def parse_annotation(self, file_data):
        file_name = re.findall("Image filename\W*[\w/.]*\W", file_data)
        file_name = re.findall("\"*[\w/.]*\"", file_name[0])
        file_name = file_name[0].replace("\"", "")

        image_size = re.findall("Image size \\(X x Y x C\\) : [0-9]* x [0-9]* x [0-9]*", file_data)
        image_size = re.findall("\d+", image_size[0])
        image_size = tuple(image_size)

        database = re.search("Database : \".*\"", file_data)
        database = re.findall("\".*\"", database.group(0))

        objects_ground_truth = re.findall("Objects with ground truth .*", file_data)
        objects_ground_truth = re.findall("\"\w*\"", objects_ground_truth[0])

        objects = []
        for object_ground_truth in set(objects_ground_truth):
            objects_match = re.findall(".*{}.*".format(object_ground_truth), file_data)
            objects = objects + self.parse_objects(objects_match)

        return ImageAnnotation(file_name, image_size, database[0], objects_ground_truth, objects)

    def parse_objects(self, objects):
        image_objects = []
        for i in range (1, len(objects), 4):
            label = objects[i+1]
            label = re.findall("(?::).*", label)
            center_point = objects[i+2]
            center_point = re.findall("(?::).*", center_point)
            center_point = re.findall("\d+", center_point[0])
            center_point = tuple(center_point)

            bounding_box = objects[i+3]
            bounding_box = re.findall("(?::).*", bounding_box)
            bounding_box = re.findall("\d+", bounding_box[0])
            bounding_box = tuple(bounding_box)

            image_objects.append(ImageObject(label[:], center_point[:], bounding_box[:]))

        return image_objects