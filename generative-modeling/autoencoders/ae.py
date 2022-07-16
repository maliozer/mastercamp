#%%
# tensorflow keras implementation

from smtpd import DebuggingServer
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

import math

#%%

def plot_data(file, num_images, images, labels):
  ''' visualization of data '''
  grid = math.ceil(math.sqrt(num_images))
  plt.figure(figsize=(grid*2,grid*2))
  for i in range(num_images):
      plt.subplot(grid,grid,i+1)
      plt.xticks([])
      plt.yticks([])
      plt.grid(False)     
      plt.imshow(images[i].reshape(28,28))
      plt.xlabel(class_names[labels[i]])      
  plt.savefig(file)


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, x_test.shape)

# Normalize the pixel values from 255 to [0,1]
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flattening the images from 28x28 to 784
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_test.shape)

class_names = ['Zero', 'One', 'Two', 'Three', 'Four',
               'Five', 'Six', 'Seven', 'Eight', 'Nine']


file = 'contents/before.png'
plot_data(file, 25, x_train, y_train)

#%%

# Build the AE

# Encoder : encodes the data into some lower representation
# Decoder : rebuild the original based on the model encoded represenation

# import abstractions from TF/Keras for building DL model

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# encoding dimension size means reduce data from 784 to 32
encoding_dim = 32

# Encoder portion
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))

# Decoder Portion : input into the decoder is the output from the encoded layer
decoded = Dense(784, activation='sigmoid')(encoded)
autoencoder = Model(input_img, decoded)
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))


# %%

# Training the AE
autoencoder.compile(optimizer='adam', loss='mse')