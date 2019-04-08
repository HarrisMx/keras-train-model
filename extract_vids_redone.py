#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:57:31 2019

@author: Mxolisi
"""

import os
import numpy as np
import csv
import subprocess
import cv2
import convert as conv

class read_csv:
    
    file_name = ''
    file_path = ''
    read_lines = []
    train_data = ''
    dataset_path = ''
    files = ''
    
    def __init__(self, file_name, path, dataset_path,train_data):
        self.file_name = file_name
        self.file_path = path
        self.train_data = train_data
        self.dataset_path = dataset_path
        self.files = os.listdir(self.dataset_path)
        return
    
    #get frames
    def getFrame(self,sec,video_file,_file_name,folder_name):
        video_file.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = video_file.read()
        if hasFrames:
            # save frame as JPG file
            cv2.imwrite(_file_name+str(sec)+".png", image)
        return hasFrames
    
    #creating folders and sorting files
    def read_files(self):
        
        with open('../csv files/' + self.file_name, newline='') as csvFile:
            