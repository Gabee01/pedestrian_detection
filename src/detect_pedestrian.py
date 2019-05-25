from parser import *
from db_loader import *
import cv2

loader = DbLoader()
board = ImagesBoard()
parser = Parser()

images_path = loader.load_databases()
annotations = loader.load_annotations()

for annotation in annotations:
    file_data = loader.read_file(annotation)
    image_annotation = parser.parse_annotation(file_data)
    print(image_annotation.__dict__)
    print(loader.databasesPath + image_annotation.file_name)
    image = cv2.imread(loader.databasesPath + image_annotation.file_name)

    print(image_annotation.objects)
    for object in image_annotation.objects:
        print(object.bounding_box)
        object_start = (int(object.bounding_box[0]), int(object.bounding_box[1]))
        object_end = (int(object.bounding_box[2]), int(object.bounding_box[3]))
        cropped_image = image[object_start[1]:object_end[1], object_start[0]:object_end[0]]
        cv2.imshow('img', image)
        cv2.imshow('cropped', cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # board.add_to_plot(loader.read_raw_image(image), [0,0], 'image')