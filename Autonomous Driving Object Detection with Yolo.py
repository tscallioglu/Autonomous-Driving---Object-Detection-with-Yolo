import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import imageio

import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model

from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body

from yolo_functions import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


print("yolo_anchors.txt:", "\n" , "0.57273, 0.677385", "\n", "1.87446, 2.06253", "\n","3.33843, 5.47434", "\n", "7.88282, 3.52778 ", "\n", "9.77052,9.16828")


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then gets rid of the corresponding box.
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes.
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes.
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes.
    
    Note: "None" is here because It isn't know the exact number of selected boxes, as it depends on the threshold. 
    """
    
    # Step 1: Computes box scores
    box_scores = box_confidence*box_class_probs

    # Step 2: Finds the box_classes using the max box_scores, keep track of the corresponding score.
    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1)

    # Step 3: Creates a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes that wanted to be kept (with probability >= threshold).
    filtering_mask = tf.greater(box_class_scores,threshold)
    
    # Step 4: Applys the mask to box_class_scores, boxes and box_classes.
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    
    return scores, boxes, classes


def iou(box1, box2):
    """
    Implements the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (box1_x1, box1_y1, box1_x2, box_1_y2)
    box2 -- second box, list object with coordinates (box2_x1, box2_y1, box2_x2, box2_y2)
    """

    # Assigns variable names to coordinates for clarity.
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
    
    # Calculates the (yi1, xi1, yi2, xi2) coordinates of the intersection of box1 and box2.
    xi1 = max(box1_x1,box2_x1)
    yi1 = max(box1_y1,box2_y1)
    xi2 = min(box1_x2,box2_x2)
    yi2 = min(box1_y2,box2_y2)
    inter_width = xi2-xi1
    inter_height = yi2-yi1
    inter_area = max(inter_width,0)*max(inter_height,0)  

    # Calculates the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))
    union_area = (box1_area+box2_area)-inter_area
    
    # Computes the IoU.
    iou = inter_area/union_area
    
    return iou


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes.
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes().
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size.
    classes -- tensor of shape (None,), output of yolo_filter_boxes().
    max_boxes -- integer, maximum number of predicted boxes that wanted to be.
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering.
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box.
    boxes -- tensor of shape (4, None), predicted box coordinates.
    classes -- tensor of shape (, None), predicted class for each box.
    """
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initializes variable max_boxes_tensor
    
    # Gets the list of indices corresponding to boxes that ate kept.
    nms_indices = tf.image.non_max_suppression(
  boxes,
  scores,
  max_boxes_tensor,
  iou_threshold=0.5,
  name=None
)
    
    # Selects only nms_indices from scores, boxes and classes.
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes



def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    # Retrieves outputs of the YOLO model.
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Converts boxes to be ready for filtering functions (converts boxes box_xy and box_wh to corner coordinates)
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Uses one of the functions that has been implemented to perform Score-filtering with a threshold of score_threshold.
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = score_threshold)
    
    # Scales boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Use one of the functions you've implemented to perform Non-max suppression with 
    # maximum number of boxes set to max_boxes and a threshold of iou_threshold.
    scores, boxes, classes =  yolo_non_max_suppression(scores, boxes, classes, max_boxes = max_boxes, iou_threshold = iou_threshold)

    return scores, boxes, classes

sess = K.get_session()

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)   

yolo_model = load_model("model_data/yolo.h5")


#yolo_model.summary()


yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))


scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)


def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the predictions.
    
    Arguments:
    sess -- tensorflow/Keras session containing the YOLO graph.
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes.
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes.
    out_classes -- tensor of shape (None, ), class index of the predicted boxes.
    
    Here "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """

    # Preprocesses your image
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Runs the session with the correct tensors and the correct placeholders in the feed_dict.
    out_scores, out_boxes, out_classes = sess.run(fetches=[scores,boxes,classes],
       feed_dict={yolo_model.input: image_data,
                  K.learning_phase():0
       })

    # Prints predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generates colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draws bounding boxes on the image file.
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Saves the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    # Displays the results.
    output_image = imageio.imread(os.path.join("out", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes


out_scores, out_boxes, out_classes = predict(sess, "test.jpg")

