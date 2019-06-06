#!/Users/johann/anaconda3/bin/python

import os
from keras.models import load_model

model = load_model('/Users/johann/github/Artificial_intelligence/Deep Learning/object detection/yolov3.weights')
model.summary()
