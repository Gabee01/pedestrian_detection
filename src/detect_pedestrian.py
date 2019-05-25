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
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # board.add_to_plot(loader.read_raw_image(image), [0,0], 'image')