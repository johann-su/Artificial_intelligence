import os
import gzip
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

# The path to the Dataset (change to your path)
digits = os.path.join("/", "Users", "johann", "github", "Artificial_intelligence", "Deep Learning", "mnist", "data", "train-images-idx3-ubyte.gz")
labels = os.path.join("/", "Users", "johann", "github", "Artificial_intelligence", "Deep Learning", "mnist", "data", "train-labels-idx1-ubyte.gz")

test_digits = os.path.join("/", "Users", "johann", "github", "Artificial_intelligence", "Deep Learning", "mnist", "data", "t10k-images-idx3-ubyte.gz")
test_labels = os.path.join("/", "Users", "johann", "github", "Artificial_intelligence", "Deep Learning", "mnist", "data", "t10k-labels-idx1-ubyte.gz")

# a function to unpack and prepare the images and labels
def open_gz_image(filename):
    with gzip.open(filename, "rb") as file:
        data = np.frombuffer(file.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28) / 255

def open_gz_label(filename):
    with gzip.open(filename, "rb") as file:
        return np.frombuffer(file.read(), np.uint8, offset=8)

# Preparing the Data
X_train = open_gz_image(digits)
y_train = open_gz_label(labels)

X_test = open_gz_image(test_digits)
y_test = open_gz_label(test_labels)

X_train = X_train.reshape(-1 ,28, 28, 1)
X_test = X_test.reshape(-1 ,28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

outcomes = y_train.shape[1]

# Training the Model
model = Sequential()

model.add(Conv2D(64, kernel_size=(5, 5), input_shape=(28, 28, 1), activation="relu"))
model.add(Conv2D(64, kernel_size=(5, 5), activation="relu"))

model.add(Conv2D(32, kernel_size=(5, 5), activation="relu"))
model.add(Conv2D(32, kernel_size=(5, 5), activation="relu"))

model.add(Flatten())

model.add(Dense(32, activation="relu"))
model.add(Dense(outcomes, activation="sigmoid"))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

model.fit(X_train, y_train, batch_size=32, epochs=8)
model.evaluate(X_test, y_test, batch_size=32)

model.save('mnist_model.h5')

# Loading the Model
model = load_model('/Users/johann/github/Artificial_intelligence/Deep Learning/mnist/mnist_model.h5')
model.summary()
