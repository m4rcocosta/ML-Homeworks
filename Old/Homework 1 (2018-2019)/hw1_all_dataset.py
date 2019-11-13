import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV

# Paths
feature_vectors_path = "./drebin/feature_vectors"
csv_path = "./drebin/sha256_family.csv"

# Feature vector
FEATURES_SET = {
    "feature": 1,
    "permission": 2,
    "activity": 3,
    "service_receiver": 3,
    "provider": 3,
    "service": 3,
    "intent": 4,
    "api_call": 5,
    "real_permission": 6,
    "call": 7,
    "url": 8
}

# Count how many features belong to a specific set
def count_feature_set(lines, features_to_considerate):
    features_map = {x: 0 for x in range(1, len(features_to_considerate.values())+1)}
    for l in lines:
        if l != "\n":
            set = l.split("::")[0]
            if set in features_to_considerate.keys():
                features_map[features_to_considerate[set]] += 1
    features = []
    for i in range(1, len(features_to_considerate.values())+1):
        features.append(features_map[i])
    return features

# Map family to category
def map_family_to_category(families):
    out = {}
    count = 1
    for family in families:
        out[family] = count
        count += 1
    return out

# Read data
def read(features_to_considerate):
    # Open .npy if they exists
    if features_to_considerate == FEATURES_SET:
        files_name = "_all"
    else:
        files_name = ""
        for elem in features_to_considerate.keys():
            files_name += "_"+elem

    if isfile("x"+files_name+".npy") and isfile("y"+files_name+".npy"):
        print "Loading previous data ..."
        x = np.load("x"+files_name+".npy")
        y = np.load("y"+files_name+".npy")
        print "-"*50
        print "Data set:"
        sys.stdout.write("x = ")
        print x.shape
        sys.stdout.write("y = ")
        print y.shape
        print "-"*50
        return x, y
    else:
        print "Reading data ..."
        dataset = [f for f in listdir(feature_vectors_path) if isfile(join(feature_vectors_path, f))]

        print "Reading csv file for ground truth ..."
        ground_truth = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=str)

        print "Reading positive and negative texts ..."
        positives = []
        negatives = []
        for elem in dataset:
            if elem in ground_truth[:, 0]:
                positives.append(elem)
            else:
                negatives.append(elem)

        print "Extracting features ..."
        x = []
        y = []
        for element in positives:
            sys.stdin = open("%s/%s" % (feature_vectors_path, element))
            features = sys.stdin.readlines()
            sample = count_feature_set(features, features_to_considerate)
            x.append(sample)
            y.append(1)

        for element in negatives:
            sys.stdin = open("%s/%s" % (feature_vectors_path, element))
            features = sys.stdin.readlines()
            sample = count_feature_set(features, features_to_considerate)
            x.append(sample)
            y.append(0)

        print "Data is read successfully!"
        x = np.array(x)
        y = np.array(y)
        print "-"*50
        print "Data set:"
        sys.stdout.write("x = ")
        print x.shape
        sys.stdout.write("y = ")
        print y.shape
        print "-"*50

        print "Saving data ..."
        print "-"*50
        np.save("x"+files_name+".npy", x)
        np.save("y"+files_name+".npy", y)

        return x, y

# Read for family classification
def read_family_classification():
    # Open .npy if they exists
    if isfile("x_family_classification.npy") and isfile("y_family_classification.npy"):
        print "Loading previous data ..."
        x = np.load("x_family_classification.npy")
        y = np.load("y_family_classification.npy")
        print "-"*50
        print "Data set:"
        sys.stdout.write("x = ")
        print x.shape
        sys.stdout.write("y = ")
        print y.shape
        print "-"*50
        return x, y
    else:
        print "Reading data ..."
        dataset = [f for f in listdir(feature_vectors_path) if isfile(join(feature_vectors_path, f))]

        print "Reading csv file for ground truth ..."
        ground_truth = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=str)
        families = np.unique(ground_truth[:, 1])
        classes = map_family_to_category(families)

        print "Reading positives texts ..."
        positives = []
        for elem in dataset:
            if elem in ground_truth[:, 0]:
                positives.append(elem)

        print "Extracting features ..."
        x = []
        y = []
        for i in range(ground_truth.shape[0]):
            sys.stdin = open("%s/%s" % (feature_vectors_path, ground_truth[i, 0]))
            features = sys.stdin.readlines()
            sample = count_feature_set(features, FEATURES_SET)
            x.append(sample)
            y.append(classes[ground_truth[i, 1]])

        print "Data is read successfully!"
        x = np.array(x)
        y = np.array(y)
        print "-"*50
        print "Data set:"
        sys.stdout.write("x = ")
        print x.shape
        sys.stdout.write("y = ")
        print y.shape
        print "-"*50

        print "Saving data ..."
        print "-"*50
        np.save("x_family_classification.npy", x)
        np.save("y_family_classification.npy", y)

        return x, y

def detect(model, family_classification, features_to_considerate):
    if model != "LR" and model != "SVM" and model != "RF":
        print "Model unknown. Possible models: LR/SVM/RF"
        exit()

    title = "Model: {}, Famyly Classification: {}, Features Vector: {}".format(model, family_classification, features_to_considerate)
    print "="*len(title)
    print title
    print "="*len(title)

    if family_classification:
        x, y = read_family_classification()
    else:
        x, y = read(features_to_considerate)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
    sys.stdout.write("x_train = ")
    print x_train.shape
    sys.stdout.write("y_train = ")
    print y_train.shape
    print "-"*50
    sys.stdout.write("x_test = ")
    print x_test.shape
    sys.stdout.write("y_test = ")
    print y_test.shape
    print "-"*50

    t0 = time()

    # Logistic Regression
    if model == "LR":
        model_name = "Logistic Regression"
        logistic_regression = LogisticRegression(solver="lbfgs", multi_class="auto")

        print "Fitting logistic regression ..."
        logistic_regression.fit(x_train, y_train)

        print "Evaluating ..."
        y_pred = logistic_regression.predict(x_test)
    
    # Support Vector Machine
    elif model == "SVM":
        model_name = "Support Vector Machine"
        svc = SVC(gamma = "scale")
     
        print "Fitting SVM ..."
        svc.fit(x_train, y_train)

        print "Evaluating ..."
        y_pred = svc.predict(x_test)

    # Random Forest
    elif model == "RF":
        model_name = "Random Forest"
        forest = RandomForestClassifier(n_estimators=100)
    
        print "Fitting RF ..."
        forest.fit(x_train, y_train)

        print "Evaluating ..."
        y_pred = forest.predict(x_test)

    print "Accuracy is %f." % accuracy_score(y_test, y_pred)
    if not family_classification:
        print "Confusion matrix:"
        print confusion_matrix(y_test, y_pred)
        print "Precision score is %f." % precision_score(y_test, y_pred)
        print "Recall score is %f." % recall_score(y_test, y_pred)
        print "F1 score is %f." % f1_score(y_test, y_pred)

    t1 = time()

    print "{} model performed in {} seconds.\n".format(model_name, round(t1-t0, 3))

features_to_considerate = {
    "permission": 1,
    "api_call": 2,
    "url": 3
}

models = ["LR", "SVM", "RF"]
feature_vectors = [FEATURES_SET, features_to_considerate]

for m in models:
    for fv in feature_vectors:
        detect(m, False, fv)

for m in models:
    detect(m, True, FEATURES_SET)