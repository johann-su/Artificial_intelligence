#!/Users/johann/anaconda3/bin/python

import PIL
from keras.models import Sequential
from keras.layers import Dense, Conv2D

input = os.path.join("images", "input", "1_1.png")
style = os.path.join("images", "input", "1_2.png")

print (input)
