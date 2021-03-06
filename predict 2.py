#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:18:42 2019

@author: Mxolisi
"""
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import argparse
import PIL
from dicttoxml import dicttoxml
import xmltodict

'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image we are going to classify")
ap.add_argument("-m", "--model", required=True,
	help="path to trained Keras model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to label binarizer")
ap.add_argument("-w", "--width", type=int, default=28,
	help="target spatial dimension width")
ap.add_argument("-e", "--height", type=int, default=28,
	help="target spatial dimension height")
ap.add_argument("-f", "--flatten", type=int, default=-1,
	help="whether or not we should flatten the image")
args = vars(ap.parse_args())
'''

model = load_model('/Users/Mxolisi/Documents/DevProjects/redone/vgg16/firsttry.h5')
input_shape = model.layers[0].output_shape[1:3]
array = []

def appendXml(data):
    
    xml = dicttoxml(array, custom_root='prediction', attr_type=False)
    wfile = open('predicted.xml','w')
    wfile.write(str(xml))
    wfile.close()
    print("**** Output data written to XML file ****")


def predict(image_path):
    # Load and resize the image using PIL.
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    # Plot the image.
    plt.imshow(img_resized)
    plt.show()

    # Convert the PIL image to a numpy-array with the proper shape.
    img_array = np.expand_dims(np.array(img_resized), axis=0)

    # Use the VGG16 model to make a prediction.
    # This outputs an array with 1000 numbers corresponding to
    # the classes of the ImageNet-dataset.
    pred = model.predict(img_array)

    # Decode the output of the VGG16 model.
    pred_decoded = decode_predictions(pred)[0]
    
    pred_name = ''
    pred_score = 0
    count = 0
    # Print the predictions.
    for code, name, score in pred_decoded:
        print("{0:>6.2%} : {1}".format(score, name))
        if(count == 0):
            pred_name = name
            pred_score = "{0:>6.2%}".format(score)
        count = count + 1
        
    img_name = image_path.split('/')[-1]
    data = {
                'Image_name': img_name,
                'class_name': pred_name,
                'score': pred_score
            }
    
    array.append(data)
    
    appendXml(array)
    
predict('/Users/Mxolisi/Documents/DevProjects/redone/animals_dataset/pour/OMO’s New Ultimate Laundry Liquid06_11.jpg')
#model.summary()