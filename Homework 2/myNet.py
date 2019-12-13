import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # only fatal errors
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import numpy as np
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
blindset = "dataset/blind"
models_dir = "models/"

def loadData():
    train_generator = train_datagen.flow_from_directory(
        directory = trainingset,
        target_size = (224, 224),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = True
    )

    test_generator = test_datagen.flow_from_directory(
        directory = testset,
        target_size = (224, 224),
        color_mode = "rgb",
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = False
    )

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
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(name + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

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
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.RMSprop(lr=1e-4), metrics = ['accuracy'])

    return model

def myNet2(input_shape, num_classes):
    
    model = Sequential()
    
    model.add(Conv2D(12, kernel_size=(11, 11), strides=(2, 2), activation='relu', input_shape=input_shape, padding="same"))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(32, kernel_size=(8, 8), strides=(2, 2), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(240, kernel_size=(8, 8), strides=(2, 2), activation='relu', padding='valid'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Conv2D(360, kernel_size=(2, 2), strides=(1, 1), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = "adam"
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
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

    try:
        history = model.fit_generator(train_generator, epochs = 100, verbose = 1, steps_per_epoch = steps_per_epoch, validation_data = test_generator, validation_steps = val_steps)
    except KeyboardInterrupt:
        pass
    end_time = time.time()
    print("Time elapsed: " + str(start_time - end_time))
    saveModel(model, model_name)
    plotHistory(history, "myNet")

def transferLearning():
    model_name = "transfer_weather"
    train_generator, test_generator = loadData()
    steps_per_epoch = train_generator.n//train_generator.batch_size
    val_steps = test_generator.n//test_generator.batch_size + 1

    start_time = time.time()

    # create the base pre-trained model
    base_model = applications.vgg16.VGG16(weights = "imagenet", include_top = False, input_tensor = Input(shape = train_generator.image_shape))

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation = 'relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(train_generator.num_classes, activation='softmax')(x)

    # this is the model we will train
    transfer_model = Model(inputs = base_model.input, outputs = predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    transfer_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])

    # train the model on the new data for a few epochs
    try:
        history_transfer = transfer_model.fit_generator(train_generator, epochs = 5, verbose = 1, steps_per_epoch = steps_per_epoch, validation_data = test_generator, validation_steps = val_steps)
    except KeyboardInterrupt:
        pass

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.
    '''
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)'''

    # we chose to train the top 2 layers, i.e. we will freeze
    # the first 18 layers and unfreeze the rest:
    for layer in transfer_model.layers[:18]:
        layer.trainable = False
    for layer in transfer_model.layers[18:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    transfer_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics = ['accuracy'])

    # we train our model again (this time fine-tuning the top 2 vgg16 blocks
    # alongside the top Dense layers
    try:
        history_transfer = transfer_model.fit_generator(train_generator, epochs = 95, verbose = 1, steps_per_epoch = steps_per_epoch, validation_data = test_generator, validation_steps = val_steps)
    except KeyboardInterrupt:
        pass
    end_time = time.time()
    print("Time elapsed: " + str(start_time - end_time))
    saveModel(transfer_model, model_name)
    plotHistory(history_transfer, "transferNet")

def test(net):
    train_generator, test_generator = loadData()
    model_name = "%s_weather" % net
    model = loadModel(model_name)
    if model == None:
        print("Model doesn't exist!")
        exit(1)
    classnames = [k for k,v in train_generator.class_indices.items()]
    evaluateModel(model, test_generator, classnames)

if __name__ == "__main__":
    tasks = {
        "train": train,
        "test": test,
        "transferlearning": transferLearning
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
    else:
        transferLearning()

   