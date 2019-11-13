import numpy as np
import os
import time
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
import tensorflow as tf

with tf.device('/gpu:0'):

	train_dir = "dataset/train_bin"
	test_dir = "dataset/test_bin"

	datagen = image.ImageDataGenerator(rescale=1./255)
	batch_size = 32

	num_classes = 2
	 
	train_generator = datagen.flow_from_directory(
	    train_dir,
	    target_size=(224, 224),
	    batch_size=batch_size,
	    class_mode='binary',
	    shuffle=True)

	test_generator = datagen.flow_from_directory(
	    test_dir,
	    target_size=(224, 224),
	    batch_size=batch_size,
	    class_mode='binary',
	    shuffle=False)

	test_generator
	train_generator

	train_samples, test_samples = train_generator.samples, test_generator.samples

	image_input = Input(shape=(224, 224, 3))

	model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
	model.summary()
	last_layer = model.get_layer('fc2').output
	#x= Flatten(name='flatten')(last_layer)
	out = Dense(num_classes, activation='softmax', name='output')(last_layer)
	custom_vgg_model = Model(image_input, out)
	custom_vgg_model.summary()

	for layer in custom_vgg_model.layers[:-1]:
		layer.trainable = False

	custom_vgg_model.layers[3].trainable

	custom_vgg_model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


	t=time.time()
	#	t = now()
	hist = custom_vgg_model.fit_generator(train_generator, 
										  epochs=12, 
										  steps_per_epoch = train_samples//batch_size,
										  validation_data = test_generator,
										  validation_steps = train_samples//batch_size)
	print('Training time: %s' % (t - time.time()))
	(loss, accuracy) = custom_vgg_model.evaluate_generator(test_generator, batch_size=10, verbose=1)

	print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


	####################################################################################################################

	#Training the feature extraction also

	image_input = Input(shape=(224, 224, 3))

	model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')

	model.summary()

	last_layer = model.get_layer('block5_pool').output
	x= Flatten(name='flatten')(last_layer)
	x = Dense(128, activation='relu', name='fc1')(x)
	x = Dense(128, activation='relu', name='fc2')(x)
	out = Dense(num_classes, activation='softmax', name='output')(x)
	custom_vgg_model2 = Model(image_input, out)
	custom_vgg_model2.summary()

	# freeze all the layers except the dense layers
	for layer in custom_vgg_model2.layers[:-3]:
		layer.trainable = False

	custom_vgg_model2.summary()

	custom_vgg_model2.compile(loss='sparse_categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

	t=time.time()
	#	t = now()
	hist = custom_vgg_model2.fit_generator(train_generator,
										   epochs=12, 
										   steps_per_epoch = train_samples//batch_size,
										   validation_data=test_generator,
										   validation_steps = test_samples//batch_size)
	print('Training time: %s' % (t - time.time()))
	(loss, accuracy) = custom_vgg_model2.evaluate(test_generator, batch_size=10, verbose=1)

	print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

	#%%
	import matplotlib.pyplot as plt
	# visualizing losses and accuracy
	train_loss=hist.history['loss']
	val_loss=hist.history['val_loss']
	train_acc=hist.history['acc']
	val_acc=hist.history['val_acc']
	xc=range(12)

	plt.figure(1,figsize=(7,5))
	plt.plot(xc,train_loss)
	plt.plot(xc,val_loss)
	plt.xlabel('num of Epochs')
	plt.ylabel('loss')
	plt.title('train_loss vs val_loss')
	plt.grid(True)
	plt.legend(['train','val'])
	#print plt.style.available # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])

	plt.figure(2,figsize=(7,5))
	plt.plot(xc,train_acc)
	plt.plot(xc,val_acc)
	plt.xlabel('num of Epochs')
	plt.ylabel('accuracy')
	plt.title('train_acc vs val_acc')
	plt.grid(True)
	plt.legend(['train','val'],loc=4)
	#print plt.style.available # use bmh, classic,ggplot for big pictures
	plt.style.use(['classic'])


