import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # only fatal errors
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import numpy as np
import cv2
import argparse
import time
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers, optimizers, applications, callbacks
from keras.models import load_model, Model, Input, Sequential
import sklearn.metrics 
from sklearn.metrics import classification_report, confusion_matrix

from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
#config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config = config)
set_session(sess) # set this TensorFlow session as the default session for Keras

trainingset = "dataset/train/"
testset = "dataset/test/"
blindset = "dataset/blind/"
models_dir = "models/"

def loadData():
    train_generator = train_datagen.flow_from_directory(
        directory = trainingset,
        target_size = (118, 224),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True
    )

    test_generator = test_datagen.flow_from_directory(directory = testset)

    num_samples = train_generator.n
    num_classes = train_generator.num_classes
    input_shape = train_generator.image_shape

    classnames = [k for k,v in train_generator.class_indices.items()]

    print("Image input %s" %str(input_shape))
    print("Classes: %r" %classnames)

    print("Loaded %d training samples from %d classes." % (num_samples, num_classes))
    print("Loaded %d test samples from %d classes." % (test_generator.n, test_generator.num_classes))

    return (train_generator, test_generator)

def loadModel(model_name):
    filename = os.path.join(models_dir, '%s.h5' %model_name)
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

def saveModel(model, model_name):
    filename = os.path.join(models_dir, '%s.h5' %model_name)
    model.save(filename)
    print("\nModel saved successfully on file %s\n" %filename)

def evaluateModel(model, test_generator, classnames):
    val_steps = test_generator.n//test_generator.batch_size + 1
    loss, acc = model.evaluate_generator(test_generator, verbose = 1, steps = val_steps)
    print('Test loss: %f' %loss)
    print('Test accuracy: %f' %acc)

    preds = model.predict_generator(test_generator,verbose=1,steps=val_steps)

    Ypred = np.argmax(preds, axis = 1)
    Ytest = test_generator.classes  # shuffle=False in test_generator

    print(classification_report(Ytest, Ypred, labels = None, target_names = classnames, digits = 3))

    cm = confusion_matrix(Ytest, Ypred)

    conf = [] # data structure for confusions: list of (i,j,cm[i][j])
    for i in range(0, cm.shape[0]):
        for j in range(0, cm.shape[1]):
            if (i != j and cm[i][j] > 0):
                conf.append([i, j, cm[i][j]])

    col = 2
    conf = np.array(conf)
    conf = conf[np.argsort(-conf[:, col])]  # decreasing order by 3-rd column (i.e., cm[i][j])

    print('%-16s     %-16s  \t%s \t%s ' %('True','Predicted','errors','err %'))
    print('------------------------------------------------------------------')
    for k in conf:
        print('%-16s ->  %-16s  \t%d \t%.2f %% ' %(classnames[k[0]], classnames[k[1]], k[2], k[2]*100.0/test_generator.n))


def plotHistory(history, name):

    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(name + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name + '_accuracy.png')
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(name + '_loss.png')

def myNet(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(16, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(lr=1e-4), metrics = ['accuracy'])

    return model

def myNet2(input_shape, num_classes):
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(num_classes, activation='softmax'))
	# compile model
	opt = optimizers.SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def alexNet(input_shape, num_classes, regl2 = 0.0001, lr = 0.0001):

    model = Sequential()

    # C1 Convolutional Layer 
    model.add(Conv2D(filters = 96, input_shape = input_shape, kernel_size = (11,11), strides = (2,4), padding = 'valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters = 256, kernel_size = (11,11), strides = (1,1), padding = 'valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C4 Convolutional Layer
    model.add(Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C5 Convolutional Layer
    model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = 'valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Flatten
    model.add(Flatten())

    flatten_shape = (input_shape[0] * input_shape[1] * input_shape[2],)
    
    # D1 Dense Layer
    model.add(Dense(4096, input_shape = flatten_shape, kernel_regularizer = regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(4096, kernel_regularizer = regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D3 Dense Layer
    model.add(Dense(1000,kernel_regularizer = regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile
    adam = optimizers.Adam(lr = lr)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model

def train(net):
    train_generator, test_generator = loadData()
    classnames = [k for k,v in train_generator.class_indices.items()]

    start_time = time.time()
    # Create the model
    model_name = "%s_weather" % net
    model = nets[net](train_generator.image_shape, train_generator.num_classes)
    model.summary()

    steps_per_epoch = train_generator.n//train_generator.batch_size
    val_steps = test_generator.n//test_generator.batch_size + 1

    history = model.fit_generator(train_generator, epochs = 3, verbose = 1, steps_per_epoch = steps_per_epoch, validation_data = test_generator, validation_steps = val_steps)
    end_time = time.time()
    print("Time elapsed: " + str(end_time - start_time))
    saveModel(model, model_name)
    plotHistory(history, net)

def transferLearning():
    model_name = "transfer_weather"
    train_generator, test_generator = loadData()
    steps_per_epoch = train_generator.n//train_generator.batch_size
    val_steps = test_generator.n//test_generator.batch_size + 1

    start_time = time.time()

    # load model
    transfer_model = applications.vgg16.VGG16(include_top = False, input_shape = train_generator.image_shape)
	# mark loaded layers as not trainable
    for layer in transfer_model.layers:
        layer.trainable = False
	# allow last vgg block to be trainable
    transfer_model.get_layer('block5_conv1').trainable = True
    transfer_model.get_layer('block5_conv2').trainable = True
    transfer_model.get_layer('block5_conv3').trainable = True
    transfer_model.get_layer('block5_pool').trainable = True
	# add new classifier layers
    flat1 = Flatten()(transfer_model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(train_generator.num_classes, activation='softmax')(class1)
	# define new model
    transfer_model = Model(inputs = transfer_model.inputs, outputs = output)
    # compile model
    opt = optimizers.SGD(lr=0.01, momentum=0.9)
    transfer_model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    history_transfer = transfer_model.fit_generator(train_generator, epochs = 100, verbose = 1, steps_per_epoch = steps_per_epoch, validation_data = test_generator, validation_steps = val_steps)
    end_time = time.time()
    print("Time elapsed: " + str(end_time - start_time))
    plotHistory(history_transfer, "transferNet")
    saveModel(transfer_model, model_name)

def test(net):
    train_generator, test_generator = loadData()
    model_name = "%s_weather" % net
    model = loadModel(model_name)
    if model == None:
        print("Model doesn't exist!")
        exit(1)
    classnames = [k for k,v in train_generator.class_indices.items()]
    evaluateModel(model, test_generator, classnames)

def blindPredict(net):
    test_generator = test_datagen.flow_from_directory(directory = testset)
    classnames = [k for k,v in test_generator.class_indices.items()]

    print("Loading blindset images")
    images = []
    for image_name in os.listdir(blindset):
        img = cv2.imread(os.path.join(blindset, image_name))
        img = cv2.resize(img, (224, 118))
        if img is not None:
            images.append(img)

    images = np.array(images, dtype="float") / 255.0

    model_name = "%s_weather" % net
    model = loadModel(model_name)
    if model == None:
        print("Model doesn't exist!")
        exit(1)

    pred = model.predict_classes(images)
    pred_labels = [classnames[i] for i in pred]
    
    predictions_file = open("blindset_predictions.csv", "w")
    for label in pred_labels:
        print(label, file = predictions_file)
    predictions_file.close()
    

def myDatasetPredict(net):
    print("Predict %s" % net)

if __name__ == "__main__":
    tasks = {
        "train": train,
        "test": test,
        "transferlearning": transferLearning,
        "blindpredict": blindPredict,
        "mydatasetpredict": myDatasetPredict
    }
    nets = {
        "alexnet": alexNet,
        "mynet": myNet,
        "mynet2": myNet2
    }

    parser = argparse.ArgumentParser(description = 'Weather image classification')
    parser.add_argument('-task', type = str, help = "Task type: " + ", ".join(tasks), required = True)
    parser.add_argument('-net', type = str, help = "Net type: " + ", ".join(nets), required = True)
    args = parser.parse_args()
    if args.task.lower() not in tasks.keys():
        print("invalid task %s" % args.task)
        exit(1)
    if args.net.lower() not in nets.keys():
        print("invalid net %s" % args.net)
        exit(1) 

    batch_size = 32

    print("Creating train generator")
    train_datagen = ImageDataGenerator(
            rescale = 1. / 255,\
            zoom_range = 0.1,\
            rotation_range = 10,\
            width_shift_range = 0.1,\
            height_shift_range = 0.1,\
            horizontal_flip = True,\
            vertical_flip = False)

    print("Creating test generator")
    test_datagen = ImageDataGenerator(rescale = 1. / 255)

    if args.task.lower() == "train":
        train(args.net.lower())
    elif args.task.lower() == "test":
        test(args.net.lower())
    elif args.task.lower() == "blindpredict":
        blindPredict(args.net.lower())
    elif args.task.lower() == "mydatasetpredict":
        myDatasetPredict(args.net.lower())
    else:
        transferLearning()

   