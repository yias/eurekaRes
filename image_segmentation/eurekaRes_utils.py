#!/usr/bin/env python3.7

import numpy as np
import cv2

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
        start_point = (boxes_coord[i][1], boxes_coord[i][0])
        end_point = (boxes_coord[i][3], boxes_coord[i][2])

        # Draw a rectangle using opence method cv2.rectangle()
        image = cv2.rectangle(image, start_point, end_point, (np.asscalar(Colors[i][0]), np.asscalar(Colors[i][1]), np.asscalar(Colors[i][2])), thickness)

    return image


def add_classes_names_to_image(image, boxes_coord, class_ids, class_names, scores, text_colors=None, thickness=1, font_size=0.5):
    """
    function to write text on an image
    """

    nb_boxes = boxes_coord.shape[0]

    if text_colors is None:
        Colors = random_colors(nb_boxes)
    else:
        Colors = text_colors

    for i in range(nb_boxes):
        coordinates = (boxes_coord[i][1], boxes_coord[i][0])
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        txt = "{} {:.3f}".format(label, score) if score else label
 
        image = cv2.putText(image, txt, coordinates, cv2.FONT_HERSHEY_SIMPLEX, font_size, (np.asscalar(Colors[i][0]), np.asscalar(Colors[i][1]), np.asscalar(Colors[i][2])), thickness, cv2.LINE_AA) 

    return image