from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, Input, Reshape, concatenate
from keras.models import Model
from keras.layers.core import RepeatVector
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf

# Get images to train the model from a folder called ImgToTrain
X = []
for filename in os.listdir('ImgToTrain/'):
    X.append(img_to_array(load_img('ImgToTrain/'+filename)))
X = np.array(X, dtype=float)
Xtrain = 1.0/255*X #24-bit RGB color space.


#Load weights into inception resnet v2 network
incep = InceptionResNetV2(weights='imagenet', include_top=True)
incep.graph = tf.get_default_graph()
embed_input = Input(shape=(1000,))

#Building the neural network encoder,fusion the output of encoder with ember_input and use that output as input to decoder.
ei = Input(shape=(256, 256, 1,))
e = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(ei)
e = Conv2D(128, (3,3), activation='relu', padding='same')(e)
e = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(e)
e = Conv2D(256, (3,3), activation='relu', padding='same')(e)
e = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(e)
e = Conv2D(512, (3,3), activation='relu', padding='same')(e)
e = Conv2D(512, (3,3), activation='relu', padding='same')(e)
e = Conv2D(256, (3,3), activation='relu', padding='same')(e)

f = RepeatVector(32 * 32)(embed_input)
f = Reshape(([32, 32, 1000]))(f)
f = concatenate([e, f], axis=3)
f = Conv2D(256, (1, 1), activation='relu', padding='same')(f)

d = Conv2D(128, (3,3), activation='relu', padding='same')(f)
d = UpSampling2D((2, 2))(d)
d = Conv2D(64, (3,3), activation='relu', padding='same')(d)
d = UpSampling2D((2, 2))(d)
d = Conv2D(32, (3,3), activation='relu', padding='same')(d)
d = Conv2D(16, (3,3), activation='relu', padding='same')(d)
d = Conv2D(2, (3, 3), activation='tanh', padding='same')(d)
d = UpSampling2D((2, 2))(d)

model = Model(inputs=[ei, embed_input], outputs=d)

#resize images to fit the model
def inception_embedding(grayscaled_rgb):
    grr = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grr.append(i)
    grr = np.array(grr)
    grr = preprocess_input(grr)
    with incep.graph.as_default():
        embed = incep.predict(grr)
    return embed

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

#Generate training data
def gen(b):
    for i in datagen.flow(Xtrain, batch_size=b):
        grayscaled_rgb = gray2rgb(rgb2gray(i))
        l_batch = rgb2lab(i)
        X = l_batch[:,:,:,0]
        X = X.reshape(X.shape+(1,))
        Y= l_batch[:,:,:,1:] / 128
        yield ([X, inception_embedding(grayscaled_rgb)], Y)


#ImgToTrain model
b=5
model.compile(optimizer='rmsprop', loss='mse')
model.fit_generator(gen(b), epochs=1, steps_per_epoch=1)

res = []
for filename in os.listdir('ImgToTest/'):
    res.append(img_to_array(load_img('ImgToTest/'+filename)))
res = np.array(res, dtype=float)
res = gray2rgb(rgb2gray(1.0/255*res))
rese = inception_embedding(res)
res = rgb2lab(1.0/255*res)[:,:,:,0]
res = res.reshape(res.shape+(1,))
output = model.predict([res, rese])
output *= 128

# Output results
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = res[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("result/img_"+str(i)+".png", lab2rgb(cur))
