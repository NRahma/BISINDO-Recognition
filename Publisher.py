#!/usr/bin/env python
# coding: utf-8

# # Tugas Besar - EL5006 Desain Aplikasi Interaktif
# 
# ## oleh Nurrahma - 23220087
# 
# 
# ## Publisher App

# In[ ]:


from __future__ import absolute_import, unicode_literals
import datetime

import cv2
import numpy as np
import sys
import time

import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from kombu import Connection
from kombu import Exchange
from kombu import Producer
from kombu import Queue


# Default RabbitMQ server URI
rabbit_url = 'amqp://guest:guest@localhost:5672//'

# Kombu Connection
conn = Connection(rabbit_url)
channel = conn.channel()

# Kombu Exchange
# - set delivery_mode to transient to prevent disk writes for faster delivery
exchange = Exchange("video-exchange", type = "direct", delivery_mode = 1)

# Kombu Producer
producer = Producer(exchange = exchange, channel = channel, routing_key = "video")

# Kombu Queue
queue = Queue(name = "video-queue", exchange = exchange, routing_key = "video") 
queue.maybe_bind(conn)
queue.declare()


def calAvg(frame, accum_weight):

    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    
    cv2.accumulateWeighted(frame, background, accum_weight)
    
def segmentHand(frame, threshold = 50):
    global background
    
    diff = cv2.absdiff(background.astype("uint8"), frame)

    _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)
    

model = tf.keras.models.load_model(r".\model\20210307_15-29_adam.h5")

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J', 10:'K', 11:'L', 12:'M',
             13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}

background = None
accum_weight = 0.5

ROI_top = 80
ROI_bottom = 280
ROI_right = 350
ROI_left = 550

    
camera = cv2.VideoCapture(0)
encode_param =[int(cv2.IMWRITE_JPEG_QUALITY),90]
num_frames =0
while True:
    ret, frame = camera.read()

    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 70:
        calAvg(gray_frame, accum_weight)
        if num_frames <= 59:

            cv2.putText(frame_copy, "FETCHING BACKGROUND... PLEASE WAIT", 
                        (10, 460), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255), 2)

    else: 
        hand = segmentHand(gray_frame)

        cv2.putText(frame_copy, "Ready for gesture... ", 
                    (10, 45), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)

        if hand is not None:
            thresholded, hand_segment = hand

            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0),1)

            thresholded = cv2.resize(thresholded, (200, 200))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))
            thresholded = thresholded / 255.0

            pred = model.predict(thresholded)
            cv2.putText(frame_copy, "Prediction: " + word_dict[np.argmax(pred)], (360, 95), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 1)
        
        else:
            cv2.putText(frame_copy, 'No hand detected!!!', (10, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 2)
            
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
    
    msg = frame_copy[ROI_top:ROI_bottom, ROI_right:ROI_left]            
    result, imgencode = cv2.imencode('.jpg', msg)
    producer.publish(imgencode.tobytes(), content_type='image/jpeg', content_encoding='binary')

    num_frames += 1

    cv2.imshow("Publisher - BISINDO Recognition", frame_copy)

    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()


# In[ ]:




