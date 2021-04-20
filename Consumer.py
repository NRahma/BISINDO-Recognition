#!/usr/bin/env python
# coding: utf-8

# # Tugas Besar - EL5006 Desain Aplikasi Interaktif
# 
# ## oleh Nurrahma - 23220087
# 
# 
# ## Consumer App

# In[ ]:


import cv2
import numpy as np
import sys
import time

import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from kombu import Connection, Exchange, Queue
from kombu.mixins import ConsumerMixin


# Default RabbitMQ server URI
rabbit_url = 'amqp://guest:guest@localhost:5672//'

# Kombu Message Consuming Worker
class Worker(ConsumerMixin):
    def __init__(self, connection, queues):
        self.connection = connection
        self.queues = queues

    def get_consumers(self, Consumer, channel):
        return [Consumer(queues=self.queues,
                         callbacks=[self.on_message],
                         accept=['image/jpeg'])]

    def on_message(self, body, message):
        size = sys.getsizeof(body) - 33     
        
        np_array = np.frombuffer(body, dtype=np.uint8)
        np_array = np_array.reshape((size, 1))

        image = cv2.imdecode(np_array, 1)     
                
        cv2.imshow("Consumer - BISINDO Recognition", image)
        cv2.waitKey(1)

        message.ack()


def run():
    exchange = Exchange("video-exchange", type="direct")
    queues = [Queue("video-queue", exchange, routing_key="video")]
    with Connection(rabbit_url, heartbeat=4) as conn:
            worker = Worker(conn, queues)
            worker.run()

run()


# In[ ]:





# In[ ]:




