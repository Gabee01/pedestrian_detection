class ImageObject:
    def __init__(self, label, center_point, bounding_box):
        self.label = label #''
        self.center_point = center_point #(0, 0)
        self.bounding_box = bounding_box #[(0, 0), (0, 0)]

    def __str__(self):
        return self.__dict__

class ImageAnnotation:
    def __init__(self, file_name, image_size, database, objects_ground_truth, objects):
        self.file_name = file_name
        self.image_size = image_size #(0,0,0)
        self.database = database #''
        self.objects_ground_truth = objects_ground_truth #[]

        self.objects = objects