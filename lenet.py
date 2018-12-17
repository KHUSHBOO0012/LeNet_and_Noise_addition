
# coding: utf-8

from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(test_images.shape)


# # Visualizing images of MNIST Datasets


image = train_images[1]
image=np.reshape(image,(28,28))
plt.imshow(image, cmap='gray_r')
plt.show()


# # LeNet Model: conv--> Relu--> MaxPooling--> Conv-->Maxpooling-->FCN-->Softmax

model = models.Sequential()
model.add(layers.Conv2D(32,(5,5),activation='relu',kernel_initializer='he_normal',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64,(3,3),activation='relu',kernel_initializer='he_normal'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()



train_images= train_images.reshape((60000,28,28,1))
train_images = train_images/255


test_images=test_images.reshape((10000,28,28,1))
test_images=test_images/255

train_labels=to_categorical(train_labels)

test_labels=to_categorical(test_labels)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])


# At first training on train_images without any noise addition

model.fit(train_images,train_labels,epochs=3,batch_size=240)


# Evaluting model on test_images without having any noise

test_loss,test_accuracy = model.evaluate(test_images,test_labels)
print(test_accuracy)


# # Adding gaussian noise to training_images

gauss_noise = np.random.randn(60000, 28, 28)
train_images2=train_images.reshape((60000,28,28))
noisy = train_images2 + gauss_noise
#plt.imshow(noisy[1], cmap='gray_r')
#plt.show()


noisy=noisy.reshape((60000,28,28,1))


# # Training on noisy images


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(noisy,train_labels,epochs=3,batch_size=64)


# # Testing model on noisy images



test_images=test_images.reshape((10000,28,28))
gauss_noise = np.random.randn(10000, 28, 28)
test_noisy = test_images + gauss_noise
test_noisy=test_noisy.reshape((10000,28,28,1))
test_loss,test_accuracy = model.evaluate(test_noisy,test_labels)
print(test_accuracy)


# Here we see that due ot addition of noise in test images its test accuracy has gone down significantly from 0.9883 to 0.8287.
