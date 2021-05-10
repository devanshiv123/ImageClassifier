import tensorflow as tf
import numpy as np
from tensorflow import keras

# For printing stuff
import matplotlib.pyplot as plt

#Load a pre-defined data-set for images
fashion_mnist=keras.datasets.fashion_mnist

#Pull out data from dataset
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

#Show data
plt.imshow(train_images[1],cmap='gray',vmin=0,vmax=255)
plt.show()

#Define our neural net structure
model=keras.Sequential([

    #input is a 28*28 image. It flattens 28*28 image to single 784*1 input layer
    keras.layers.Flatten(input_shape=(28,28)),

    #hidden layer 128 deep . relu returns the value or 0(works good enough,much faster)
    keras.layers.Dense(units=128,activation=tf.nn.relu),

    #output is 0-10(Depending upon what piece of clothing it is). return maxm
    keras.layers.Dense(units=10,activation=tf.nn.softmax)
])

#Compile our model
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Train ur model using training data
model.fit(train_images,train_labels,epochs=5)

#Test our model, using our testing data
test_loss=model.evaluate(test_images,test_labels)

#Make predictions
predictions=model.predict(test_images)

#Print out predictions
print(list(predictions[1]).index(max(predictions[1])))
print(predictions[1])
#Print out correct answer
print(test_labels[1])