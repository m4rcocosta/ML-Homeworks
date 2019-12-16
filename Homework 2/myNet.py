import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # only fatal errors
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import argparse
import time
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers, optimizers, applications
from keras.models import load_model, Model, Input, Sequential
import sklearn.metrics 
from sklearn.metrics import classification_report, confusion_matrix

from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.compat.v1.Session(config = config)
set_session(sess) # set this TensorFlow session as the default session for Keras

dataset_path = "dataset/"
trainingset_path = dataset_path + "MWI-Dataset/"
testset_path = dataset_path + "SMART-I_WeatherTestSet/"
blindset_path = dataset_path + "WeatherBlindTestSet/"
kucoDataset_path = dataset_path + "KucoDataset/"
models_dir = "models/"

width = 224
height = 118
epochs = 100

def loadData():
    train_generator = datagen.flow_from_directory(
        directory = trainingset_path,
        target_size = (height, width),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        subset = "training"
    )

    test_generator = datagen.flow_from_directory(
        directory = trainingset_path,
        target_size = (height, width),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        subset = "validation"
    )

    num_samples = train_generator.n
    num_classes = train_generator.num_classes
    input_shape = train_generator.image_shape

    classnames = [k for k,v in train_generator.class_indices.items()]

    print("Image input %s" %str(input_shape))
    print("Classes: %r" %classnames)

    print("Loaded %d training samples from %d classes." % (num_samples, num_classes))
    print("Loaded %d test samples from %d classes." % (test_generator.n, test_generator.num_classes))

    '''
    """##Show *n* random images"""
    n = 3
    x,y = train_generator.next()
    # x,y size is train_generator.batch_size

    for i in range(0,n):
        image = x[i]
        label = y[i].argmax()  # categorical from one-hot-encoding
        print(classnames[label])
        plt.imshow(image)
        plt.show()'''

    return (train_generator, test_generator)

def loadModel(model_name):
    filename = os.path.join(models_dir, "%s.h5" %model_name)
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

def saveModel(model, model_name):
    filename = os.path.join(models_dir, "%s.h5" %model_name)
    model.save(filename)
    print("\nModel saved successfully on file %s\n" %filename)

def evaluateModel(model, test_generator, classnames):
    val_steps = test_generator.n//test_generator.batch_size + 1
    loss, acc = model.evaluate_generator(test_generator, verbose = 1, steps = val_steps)
    print("Test loss: %f" %loss)
    print("Test accuracy: %f" %acc)

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

    print("%-16s     %-16s  \t%s \t%s " %("True","Predicted","errors","err %"))
    print("------------------------------------------------------------------")
    for k in conf:
        print('%-16s ->  %-16s  \t%d \t%.2f %% ' %(classnames[k[0]], classnames[k[1]], k[2], k[2]*100.0/test_generator.n))


def plotHistory(history, name):

    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title(name + " accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc = "upper left")
    plt.savefig("images/" + name + "_%s_epochs_accuracy.png" % epochs)
    # summarize history for loss
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title(name + " loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc = "upper left")
    plt.savefig("images/" + name + "_%s_epochs_loss.png" % epochs)

def kucoNet(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size = (5, 5), activation = "relu", input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    model.add(Dropout(0.3))
    
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model

def kucoNet2(input_shape, num_classes):
    model = Sequential()
    
    #First Convolutional layer
    model.add(Conv2D(filters = 56, kernel_size = (3,3), activation = "relu", input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #second Convolutional layer
    model.add(Conv2D(32,(3,3), activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Flattening
    model.add(Flatten())

    #Hidden Layer
    model.add(Dense(units = 64, activation = "relu"))

    #Output Layer
    model.add(Dense(num_classes, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def kucoNet3(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size = (3,3), border_mode = "same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Conv2D(64, kernel_size = (2,2), border_mode = "same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), dim_ordering="th"))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = "softmax"))

    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def alexNet(input_shape, num_classes, regl2 = 0.0001, lr = 0.0001):

    model = Sequential()

    # C1 Convolutional Layer 
    model.add(Conv2D(filters = 96, input_shape = input_shape, kernel_size = (11,11), strides = (2,4), padding = "valid"))
    model.add(Activation("relu"))
    # Pooling
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "valid"))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters = 256, kernel_size = (11,11), strides = (1,1), padding = "valid"))
    model.add(Activation("relu"))
    # Pooling
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "valid"))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), padding = "valid"))
    model.add(Activation("relu"))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C4 Convolutional Layer
    model.add(Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), padding = "valid"))
    model.add(Activation("relu"))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C5 Convolutional Layer
    model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), padding = "valid"))
    model.add(Activation("relu"))
    # Pooling
    model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2), padding = "valid"))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Flatten
    model.add(Flatten())

    flatten_shape = (input_shape[0] * input_shape[1] * input_shape[2],)
    
    # D1 Dense Layer
    model.add(Dense(4096, input_shape = flatten_shape, kernel_regularizer = regularizers.l2(regl2)))
    model.add(Activation("relu"))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(4096, kernel_regularizer = regularizers.l2(regl2)))
    model.add(Activation("relu"))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D3 Dense Layer
    model.add(Dense(1000,kernel_regularizer = regularizers.l2(regl2)))
    model.add(Activation("relu"))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))

    # Compile
    adam = optimizers.Adam(lr = lr)
    model.compile(loss = "categorical_crossentropy", optimizer = adam, metrics = ["accuracy"])

    return model

def train(net):
    train_generator, test_generator = loadData()
    classnames = [k for k,v in train_generator.class_indices.items()]

    start_time = time.time()
    # Create the model
    model_name = "%s_MWI-Dataset_%s_epochs" % (net, epochs)
    model = nets[net](train_generator.image_shape, train_generator.num_classes)
    model.summary()

    steps_per_epoch = train_generator.n//train_generator.batch_size
    val_steps = test_generator.n//test_generator.batch_size + 1

    history = model.fit_generator(train_generator, epochs = epochs, verbose = 1, steps_per_epoch = steps_per_epoch, validation_data = test_generator, validation_steps = val_steps)
    end_time = time.time()
    print("Time elapsed: " + str(end_time - start_time))
    saveModel(model, model_name)
    plotHistory(history, net)

def transferLearning():
    model_name = "transfernet_MWI-Dataset_%s_epochs" % epochs
    train_generator, test_generator = loadData()
    steps_per_epoch = train_generator.n//train_generator.batch_size
    val_steps = test_generator.n//test_generator.batch_size + 1

    start_time = time.time()

    model_vgg = applications.vgg16.VGG16(weights = "imagenet", include_top = False, input_shape = train_generator.image_shape)
 
    # Freeze the layers except the last 9 layers
    for layer in model_vgg.layers[:-9]:
        layer.trainable = False
    
    # Check the trainable status of the individual layers
    for layer in model_vgg.layers:
        print(layer, layer.trainable)

    model_transfer = Sequential()
    model_transfer.add(model_vgg)
    model_transfer.add(GlobalAveragePooling2D())
    model_transfer.add(Dropout(0.2))
    model_transfer.add(Dense(100, activation = "relu"))
    model_transfer.add(Dense(train_generator.num_classes, activation="softmax"))
    model_transfer.summary()
    opt = optimizers.Adam(lr = 0.00001)
    model_transfer.compile(loss = "categorical_crossentropy", optimizer = opt,metrics = ["accuracy"])
    model_transfer.summary()

    history_transfer = model_transfer.fit_generator(train_generator, epochs = epochs, verbose = 1, steps_per_epoch = steps_per_epoch, validation_data = test_generator, validation_steps = val_steps)

    end_time = time.time()
    print("Time elapsed: " + str(end_time - start_time))
    plotHistory(history_transfer, "transferNet")
    saveModel(model_transfer, model_name)

def test(net):
    datagen = ImageDataGenerator(rescale = 1. / 255)
    test_generator = datagen.flow_from_directory(directory = testset_path,
        target_size = (height, width),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True)
    model_name = "%s_MWI-Dataset_%s_epochs" % (net, epochs)
    model = loadModel(model_name)
    if model == None:
        print("Model doesn't exist!")
        exit(1)
    classnames = [k for k,v in test_generator.class_indices.items()]
    evaluateModel(model, test_generator, classnames)

def blindPredict(net):
    generator = datagen.flow_from_directory(directory = trainingset_path)
    classnames = [k for k,v in generator.class_indices.items()]

    print("Loading blindset_path images")
    image_names = []
    for image_name in os.listdir(blindset_path):
        image_names.append(image_name)
    image_names.sort() # Images in alphabetical order

    images = []
    for image_name in image_names:
        img = cv2.imread(os.path.join(blindset_path, image_name))
        img = cv2.resize(img, (width, height))
        if img is not None:
            images.append(img)

    images = np.array(images, dtype = "float") / 255.0

    model_name = "%s_MWI-Dataset_%s_epochs" % (net, epochs)
    model = loadModel(model_name)
    if model == None:
        print("Model doesn't exist!")
        exit(1)

    pred = model.predict_classes(images)
    pred_labels = [classnames[i] for i in pred]

    counters = {
        "SUNNY": 0,
        "SNOWY": 0,
        "HAZE": 0,
        "RAINY": 0
    }
    
    predictions_file = open("WeatherBlindTestSet_predictions.csv", "w")
    for label in pred_labels:
        print(label, file = predictions_file)
        counters[label] += 1
    predictions_file.close()
    print(counters)
    

def kucoDatasetTest(net):
    datagen = ImageDataGenerator(rescale = 1. / 255)
    test_generator = datagen.flow_from_directory(directory = kucoDataset_path,
        target_size = (height, width),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True)
    model_name = "%s_MWI-Dataset_%s_epochs" % (net, epochs)
    model = loadModel(model_name)
    if model == None:
        print("Model doesn't exist!")
        exit(1)
    classnames = [k for k,v in test_generator.class_indices.items()]
    evaluateModel(model, test_generator, classnames)

if __name__ == "__main__":
    tasks = {
        "train": train,
        "test": test,
        "transferlearning": transferLearning,
        "blindpredict": blindPredict,
        "kucodatasettest": kucoDatasetTest
    }
    nets = {
        "alexnet": alexNet,
        "kuconet": kucoNet,
        "kuconet2": kucoNet2,
        "kuconet3": kucoNet3,
        "transfernet": None
    }

    parser = argparse.ArgumentParser(description = "Weather image classification")
    parser.add_argument("-task", type = str, help = "Task type: " + ", ".join(tasks), required = True)
    parser.add_argument("-net", type = str, help = "Net type: " + ", ".join(nets), required = True)
    args = parser.parse_args()
    if args.task.lower() not in tasks.keys():
        print("invalid task %s" % args.task)
        exit(1)
    if args.net.lower() not in nets.keys():
        print("invalid net %s" % args.net)
        exit(1) 

    batch_size = 32

    print("Creating dataset generator")
    datagen = ImageDataGenerator(
        rescale = 1. / 255, \
        validation_split = 0.2, \
        rotation_range = 40, \
        zoom_range = 0.2, \
        width_shift_range = 0.2, \
        height_shift_range = 0.2, \
        horizontal_flip = True, \
        vertical_flip = True)

    if args.task.lower() == "train":
        if args.net.lower() == "transfernet":
            print("Wrong net")
            exit(1)
        print("Training %s..." % args.net.lower())
        train(args.net.lower())
    elif args.task.lower() == "test":
        print("Testing %s..." % args.net.lower())
        test(args.net.lower())
    elif args.task.lower() == "blindpredict":
        print("Blind test predictions with %s..." % args.net.lower())
        blindPredict(args.net.lower())
    elif args.task.lower() == "kucodatasettest":
        print("My dataset predictions with %s..." % args.net.lower())
        kucoDatasetTest(args.net.lower())
    elif args.task.lower() == "transferlearning":
        print("Transfer Learning...")
        transferLearning()

   