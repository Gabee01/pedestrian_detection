from parser import *
from db_loader import *
from pyramid import *
# from images_board import *
from hog import *

loader = DbLoader()
# board = ImagesBoard()
parser = Parser()
pyramid = Pyramid()
hog = Hog()

images_path = loader.load_databases()
annotations = loader.load_annotations()

# def print_list(img_list, img_name):
#     # show the resized image
#     for image in img_list:
#         cv2.imshow(img_name + str(image.shape), image)

#     cv2.destroyAllWindows()

for annotation in annotations:
    file_data = loader.read_file(annotation)
    image_annotation = parser.parse_annotation(file_data)
    print(image_annotation.__dict__)
    print(loader.databasesPath + image_annotation.file_name)
    original_image = cv2.imread(loader.databasesPath + image_annotation.file_name)

    # print(image_annotation.objects)
    for image_object in image_annotation.objects:
        # print(image_object.bounding_box)
        object_start = (int(image_object.bounding_box[0]), int(image_object.bounding_box[1]))
        object_end = (int(image_object.bounding_box[2]), int(image_object.bounding_box[3]))
        cropped_image = original_image[object_start[1]:object_end[1], object_start[0]:object_end[0]]
        resized_image = cv2.resize(cropped_image, (64, 128))
        pyramids = pyramid.get_pyramids(resized_image)
        # print_list(pyramids, image_annotation.file_name)

        images_hogs = []        
        # for image in pyramids:
        mag, angle = hog.compute(pyramids[0])
            # images_hogs.append(angle)
            # images_hogs.append(mag)
        # print_list(images_hogs, image_annotation.file_name)

    # board.add_to_ot(loader.read_raw_image(image), [0,0], 'image')