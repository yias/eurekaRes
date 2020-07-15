#!/usr/bin/env python3.7

from distutils.version import LooseVersion
import numpy as np
import cv2
import tensorflow as tf
import skimage.transform
import skimage.io as io


def random_colors(cSize):
    """
    create color pallet, matrix cSize x 3
    """
    rgbl = np.array([], dtype=int).reshape(0, 3)
    for _ in range(cSize):
        rgbl = np.vstack((rgbl, np.random.randint(low=0, high=255, size=3)))

    return rgbl

def draw_boxes(image, boxes_coord, color_pallete=None, thickness=2):
    """
    draw rectancles on an images
    """
    nb_boxes = boxes_coord.shape[0]

    if color_pallete is None:
        Colors = random_colors(nb_boxes)
    else:
        Colors = color_pallete
    

    for i in range(nb_boxes):
        # left = 
        # start_point = (int(boxes_coord[i][1]), int(boxes_coord[i][0]))
        # end_point = (int(boxes_coord[i][3]), int(boxes_coord[i][2]))
        start_point = (int(boxes_coord[i][1]), int(boxes_coord[i][0]))
        end_point = (int(boxes_coord[i][3]), int(boxes_coord[i][2]))
        # print("start_point type: ", type(start_point))
        # print(start_point)
        # print(boxes_coord[i][1])
        # print(type(boxes_coord[i][1]))

        # print("end_point type: ", type(end_point))
        # Draw a rectangle using opence method cv2.rectangle()
        image = cv2.rectangle(image, start_point, end_point, (np.asscalar(Colors[i][0]), np.asscalar(Colors[i][1]), np.asscalar(Colors[i][2])), thickness)
        

    return image


def add_classes_names_to_image(image, boxes_coord, class_names, scores, text_colors=None, thickness=2, font_size=1):
    """
    function to write text on an image
    """

    nb_boxes = boxes_coord.shape[0]

    if text_colors is None:
        Colors = random_colors(nb_boxes)
    else:
        Colors = text_colors

    for i in range(nb_boxes):
        # left = (boxes_coord[i][0] - boxes_coord[i][2]/2.0)*image.shape[1]
        # top = (boxes_coord[i][1] - boxes_coord[i][3]/2.0)*image.shape[0]
        coordinates = (int(boxes_coord[i][1]), int(boxes_coord[i][0]))
        # coordinates = (int(left), int(top))
        # class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[i]
        txt = "{} {:.3f}".format(label, score) if score else label

        image = cv2.putText(image, txt, coordinates, cv2.FONT_HERSHEY_SIMPLEX, font_size, (np.asscalar(Colors[i][0]), np.asscalar(Colors[i][1]), np.asscalar(Colors[i][2])), thickness, cv2.LINE_AA) 

    return image


def preprocess_image(image, bbox, fHeight, fWidth, do_padding, min_scale=None):
    """
    funtion to process the image before training or testing

    it returns the pre-processed image, the new annotations, the window size, the scale and the padding area
    """
    new_img, window, scale, padding = mold_image(image, fHeight, fWidth, do_padding)

    if bbox is not None:
        bbox = mold_ann_boxes(bbox, scale, padding, do_padding)
        # bbox = bbox*scale
        # if do_padding:
        #     bbox[:, 1] = bbox[:, 1] + padding[0][0]
        #     bbox[:, 0] = bbox[:, 0] + padding[1][0]

    return new_img, bbox, window, scale, padding


def mold_ann_boxes(bbox, scale, padding, do_padding):
    """
    function to mold the annotation boxes
    """
    bbox = bbox*scale
    if do_padding:
        bbox[:, 1] = bbox[:, 1] + padding[0][0]
        bbox[:, 0] = bbox[:, 0] + padding[1][0]

    return bbox


def find_image_molding(image_shape, fHeight, fWidth, do_padding, min_scale=None):
    """
    Function to find how the image should be molded according to the final image shape

    It returns the window where the original image exist, the scalling
    and the padding that needs to be done on the image
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image_shape

    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    # crop = None

    final_image_shape = [fHeight, fWidth]
    max_dim = np.argmax(image_shape)
    # Scale up but not down
    scale = final_image_shape[max_dim]/image_shape[max_dim]

    # Does it exceed max dim?
    if round(scale*image_shape[1-max_dim]) > final_image_shape[1-max_dim]:
        scale = final_image_shape[1-max_dim]/image_shape[1-max_dim]

    # Need padding or cropping?
    if do_padding:
        # Get new height and width
        h = round(h*scale)
        w = round(w*scale)
        top_pad = (fHeight - h) // 2
        bottom_pad = fHeight - h - top_pad
        left_pad = (fWidth - w) // 2
        right_pad = fWidth - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        window = (top_pad, left_pad, h + top_pad, w + left_pad)


    # print("inside resize image, final image shape: ", image.shape)
    return window, scale, padding


def fast_mold_image(image, scale, padding, do_padding):
    """
    function to mold the image accoding to the scaling and padding preferences

    it returns the molded image
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype

    # Default window
    h, w = image.shape[:2]

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

    if do_padding:
        image = np.pad(image, padding, mode='constant', constant_values=0)

    return image.astype(image_dtype)


def mold_image(image, fHeight, fWidth, do_padding, min_scale=None):
    """
    function to resize and add padding in the image
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype

    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]

    # if the image is in gray scale, append the extend 2D matrix to 3D by repeating the values of the image
    if len(image.shape) < 3:
        tmp_image = np.dstack((image, image))
        image = np.dstack((tmp_image, image))

    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    # crop = None

    final_image_shape = [fHeight, fWidth]
    max_dim = np.argmax(image.shape)
    # Scale up but not down
    scale = final_image_shape[max_dim]/image.shape[max_dim]

    # Does it exceed max dim?
    if round(scale*image.shape[1-max_dim]) > final_image_shape[1-max_dim]:
        scale = final_image_shape[1-max_dim]/image.shape[1-max_dim]

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)), preserve_range=True)

    # if bbox is not None:
    #     bbox = bbox*scale

    # Need padding or cropping?
    if do_padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (fHeight - h) // 2
        bottom_pad = fHeight - h - top_pad
        left_pad = (fWidth - w) // 2
        right_pad = fWidth - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
        # if bbox is not None:
        #     bbox[:, 1] = bbox[:, 1] + top_pad
        #     bbox[:, 0] = bbox[:, 0] + left_pad


    # print("inside resize image, final image shape: ", image.shape)
    return image.astype(image_dtype), window, scale, padding


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def process_data_mine(data_slice, t_data_dir):
    """
    a function to read and mold the input image and return the image with the corresponding label
    it's slower than tf.image.resize_with_pad
    """

    # load image
    img = io.imread(t_data_dir + "/" + data_slice.get('filename'))

    # load the scale and the padding that need for molding the image
    scale = data_slice.get('scale')
    padding = data_slice.get('padding')
    padding = eval('[' + padding + ']')[0]

    # mold image
    img = fast_mold_image(img, scale, padding, True)

    # set the label
    label = [data_slice.get('class_label'), data_slice.get('xmin'), data_slice.get('ymin'), data_slice.get('xmax'), data_slice.get('ymax')]

    # return the image and its correspoding label
    return img, label


def process_data2(data_slice, t_data_dir):
    """
    a function to read and mold the input image and return the image with the corresponding label
    """

    # get the image file
    img = tf.io.read_file(t_data_dir + "/" + data_slice.get('filename'))

    # load the image
    img = tf.image.decode_jpeg(img, channels=3)

    # mold the image
    img = tf.image.resize_with_pad(img, 720, 1280)

    # get class "binary" class id
    class_id = data_slice.get('class_id')
    class_id = class_id.strip('][').split(' ')
    class_id = list(map(np.float64, class_id))


    # set the label
    label = class_id + [data_slice.get('xmin'), data_slice.get('ymin'), data_slice.get('xmax'), data_slice.get('ymax')]

    # return the image and its correspoding label
    return img, label


def get_cm(boxes_coord):
    """
    function to get the center of mass, the length on each dimension of the box and its area 
    """

    # number of boxes
    nb_boxes = boxes_coord.shape[0]

    # the center of mass
    mass_center = np.array([]).reshape((0, 2))

    # area of the boxes
    bx_area = np.array([])

    # the length of each dimension
    dims_size = np.array([]).reshape((0, 2))

    for i in range(nb_boxes):
        mass_center = np.vstack([mass_center, np.array([(boxes_coord[i][1] + boxes_coord[i][3])/2, boxes_coord[i][2]])])
        dims_size = np.vstack([dims_size, np.array([boxes_coord[i][3] - boxes_coord[i][1], boxes_coord[i][2] - boxes_coord[i][0]])])
        bx_area = np.hstack([bx_area, dims_size[i, 0] * dims_size[i, 1]])
        # start_point = (int(boxes_coord[i][1]), int(boxes_coord[i][0]))
        # end_point = (int(boxes_coord[i][3]), int(boxes_coord[i][2]))
        # print('start point: ', start_point)
        # print('end point: ', end_point)
        # print('center of mass: ', mass_center[i])
        # print('dimensions: ', dims_size[i])
        # print('area: ', bx_area[i])

    return mass_center, dims_size, bx_area


def display_real_coord(image, boxes_coord, real_coord, text_colors=None, thickness=2, font_size=0.7):
    """
    function to write text on an image
    """

    nb_boxes = boxes_coord.shape[0]

    if text_colors is None:
        Colors = random_colors(nb_boxes)
    else:
        Colors = text_colors

    for i in range(nb_boxes):
        coordinates = (int(boxes_coord[i][1] - 20), int(boxes_coord[i][2] + 15))
        # coordinates = (int(left), int(top))
        # class_id = class_ids[i]
        txt = "({:.3f}, {:.3f})".format(real_coord[i, 0], real_coord[i, 1])

        image = cv2.putText(image, txt, coordinates, cv2.FONT_HERSHEY_SIMPLEX, font_size, (np.asscalar(Colors[i][0]), np.asscalar(Colors[i][1]), np.asscalar(Colors[i][2])), thickness, cv2.LINE_AA) 

    return image


def process_bboxes(bboxes):
    """
    function to compute the proper x and y of the bounding box
    """
    # print(type(bboxes))
    bboxes[:, 0] = bboxes[:, 0] - (bboxes[:, 2]/2)
    bboxes[:, 1] = bboxes[:, 1] - (bboxes[:, 3]/2)
    bboxes[:, 2] = bboxes[:, 2] + bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] + bboxes[:, 1]
    bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
    bboxes[:, [2, 3]] = bboxes[:, [3, 2]]
    # bboxes[:, [0, 1]] = bboxes[:, [1, 0]]
    # print(bboxes.shape)
    # ttp = np.array([]).reshape(bboxes.shape)
    # print(ttp.shape)
    # ttp[:, 0] = bboxes[:, 1]
    # ttp[:, 1] = bboxes[:, 0]
    # ttp[:, 2] = bboxes[:, 3]
    # ttp[:, 3] = bboxes[:, 2]

    return bboxes
