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

trainingset = "dataset/train_big/"
testset = "dataset/test/"
blindset = "dataset/blind/"
models_dir = "models/"

width = 224
height = 118

def loadData():
    train_generator = datagen.flow_from_directory(
        directory = trainingset,
        target_size = (height, width),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        subset = "training"
    )

    test_generator = datagen.flow_from_directory(
        directory = trainingset,
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

    """##Show *n* random images"""
    n = 3
    x,y = train_generator.next()
    # x,y size is train_generator.batch_size

    for i in range(0,n):
        image = x[i]
        label = y[i].argmax()  # categorical from one-hot-encoding
        print(classnames[label])
        plt.imshow(image)
        plt.show()

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
    plt.savefig("images/" + name + '_accuracy.png')
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("images/" + name + '_loss.png')

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
    
    #First Convolutional layer
    model.add(Conv2D(filters = 56, kernel_size = (3,3), activation = 'relu', input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #second Convolutional layer
    model.add(Conv2D(32,(3,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))

    #Flattening
    model.add(Flatten())

    #Hidden Layer
    model.add(Dense(units = 64, activation = 'relu'))

    #Output Layer
    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def myNet3(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape = input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
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

    history = model.fit_generator(train_generator, epochs = 50, verbose = 1, steps_per_epoch = steps_per_epoch, validation_data = test_generator, validation_steps = val_steps)
    end_time = time.time()
    print("Time elapsed: " + str(end_time - start_time))
    saveModel(model, model_name)
    plotHistory(history, net)
'''
def transferLearning():
    model_name = "transfer_weather"
    train_generator, test_generator = loadData()
    steps_per_epoch = train_generator.n//train_generator.batch_size
    val_steps = test_generator.n//test_generator.batch_size + 1

    start_time = time.time()

    # load the pre-trained model
    # define input tensor
    input0 = Input(shape = train_generator.image_shape)

    # load a pretrained model on imagenet without the final dense layer
    feature_extractor = applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor=input0)
    
    
    feature_extractor = feature_extractor.output
    feature_extractor = Model(input=input0, output=feature_extractor)
    optimizer = 'adam' #alternative 'SGD'

    feature_extractor.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    feature_extractor.summary()


    # choose the layer from which you can get the features (block5_pool the end, glob_pooling to get the pooled version of the output)
    name_output_extractor = "block5_pool"
    trainable_layers = ["block5_conv3"]

    # build the transfer model
    # get the original input layer tensor
    input_t = feature_extractor.get_layer(index=0).input

    # set the feture extractor layers as non-trainable
    for idx,layer in enumerate(feature_extractor.layers):
        if layer.name in trainable_layers:
            layer.trainable = True
        else:
            layer.trainable = False

    # get the output tensor from a layer of the feature extractor
    output_extractor = feature_extractor.get_layer(name = name_output_extractor).output
    
    #output_extractor = MaxPooling2D(pool_size=(4,4))(output_extractor)

    # flat the output of a Conv layer
    flatten = Flatten()(output_extractor) 
    flatten_norm = BatchNormalization()(flatten)

    # add a Dense layer
    dense = Dropout(0.4)(flatten_norm)
    dense = Dense(200, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    
    # add a Dense layer
    dense = Dropout(0.4)(dense)
    dense = Dense(100, activation='relu')(dense)
    dense = BatchNormalization()(dense)

    # add the final output layer
    dense = BatchNormalization()(dense)
    dense = Dense(train_generator.num_classes, activation='softmax')(dense)
    

    model = Model(input=input_t, output=dense, name="transferNet")
    
    optimizer = 'adam' #alternative 'SGD'
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    # fit the transferNet on the training data
    stopping = callbacks.EarlyStopping(monitor='val_acc', patience=3)

    steps_per_epoch = train_generator.n//train_generator.batch_size
    val_steps = test_generator.n//test_generator.batch_size+1


    history_transfer = model.fit_generator(train_generator, epochs=50, verbose=1, callbacks=[stopping],\
                    steps_per_epoch=steps_per_epoch,\
                    validation_data=test_generator,\
                    validation_steps=val_steps)

    end_time = time.time()
    print("Time elapsed: " + str(end_time - start_time))
    plotHistory(history_transfer, "transferNet")
    saveModel(model, model_name)'''

def transferLearning():
    model_name = "transfer_weather"
    train_generator, test_generator = loadData()
    steps_per_epoch = train_generator.n//train_generator.batch_size
    val_steps = test_generator.n//test_generator.batch_size + 1

    start_time = time.time()

    model_vgg = applications.vgg16.VGG16(weights="imagenet", include_top=False, input_shape = train_generator.image_shape)
 
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
    model_transfer.add(Dense(100, activation='relu'))
    model_transfer.add(Dense(train_generator.num_classes, activation='softmax'))
    model_transfer.summary()
    opt = optimizers.Adam(lr=0.00001)
    model_transfer.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

    history_transfer = model_transfer.fit_generator(train_generator, epochs=50, verbose=1, \
                    steps_per_epoch=steps_per_epoch,\
                    validation_data=test_generator,\
                    validation_steps=val_steps)

    end_time = time.time()
    print("Time elapsed: " + str(end_time - start_time))
    plotHistory(history_transfer, "transferNet")
    saveModel(model_transfer, model_name)

def test(net):
    datagen = ImageDataGenerator(rescale = 1. / 255)
    test_generator = datagen.flow_from_directory(directory = testset,
        target_size = (height, width),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True,
        subset = "validation")
    model_name = "%s_weather" % net
    model = loadModel(model_name)
    if model == None:
        print("Model doesn't exist!")
        exit(1)
    classnames = [k for k,v in test_generator.class_indices.items()]
    evaluateModel(model, test_generator, classnames)

def blindPredict(net):
    generator = datagen.flow_from_directory(directory = trainingset)
    classnames = [k for k,v in generator.class_indices.items()]

    print("Loading blindset images")
    images = []
    for image_name in os.listdir(blindset):
        img = cv2.imread(os.path.join(blindset, image_name))
        img = cv2.resize(img, (width, height))
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
        "mynet2": myNet2,
        "mynet3": myNet3
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
        print("Training %s..." % args.net.lower())
        train(args.net.lower())
    elif args.task.lower() == "test":
        print("Testing %s..." % args.net.lower())
        test(args.net.lower())
    elif args.task.lower() == "blindpredict":
        print("Blind test predictions with %s..." % args.net.lower())
        blindPredict(args.net.lower())
    elif args.task.lower() == "mydatasetpredict":
        print("My dataset predictions with %s..." % args.net.lower())
        myDatasetPredict(args.net.lower())
    elif args.task.lower() == "transferlearning":
        print("Transfer Learning...")
        transferLearning()

   