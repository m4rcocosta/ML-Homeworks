import json
import sys
from os.path import join
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

train_dataset_path = "./dataset/train_dataset.jsonl"
test_dataset_path = "./dataset/test_dataset_blind.jsonl"

result = {}

# Read data
def read_data():
    print("Reading data ...")

    # Read train dataset
    print("Reading train dataset ...")
    dataset = []
    with open(train_dataset_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        instructions = result["instructions"]
        mnemonics = []
        for instruction in instructions:
            mnemonics.append(instruction.split(" ")[0])
        result["instructions"] = ",".join(mnemonics)
        dataset.append(result)

    train_dataset = pd.DataFrame(dataset)

    # Read blind dataset
    print("Reading test dataset ...")
    dataset = []
    with open(test_dataset_path, 'r') as json_file:
        json_list = list(json_file)
    for json_str in json_list:
        result = json.loads(json_str)
        instructions = result["instructions"]
        mnemonics = []
        for instruction in instructions:
            mnemonics.append(instruction.split(" ")[0])
        result["instructions"] = ",".join(mnemonics)
        dataset.append(result)

    test_dataset = pd.DataFrame(dataset)
    
    print("Data read successfully!")
    print("-"*50)
    sys.stdout.write("Train dataset shape: ")
    print(train_dataset.shape)
    print("-"*50)
    sys.stdout.write("Test dataset shape: ")
    print(test_dataset.shape)
    print("-"*50)
    return train_dataset, test_dataset

#split dataset
def split_dataset(classifier_type, encoder):
    target = ""
    if classifier_type == "Binary":
        target = "opt"
    else:
        target = "compiler"
    x_train, x_test, y_train, y_test = train_test_split(train_dataset["instructions"], train_dataset[target], test_size = 0.20)
    y_train = encoder.fit_transform(y_train)
    sys.stdout.write("x_train = ")
    print(x_train.shape)
    sys.stdout.write("y_train = ")
    print(y_train.shape)
    print("-"*50)
    return x_train, y_train

#classify
def classify(x_train_count, y_train, x_to_predict, classifier, classifier_type, encoder):
    print("="*50)
    print("[Classifier type: " + classifier_type + "]")
    print("="*50)
    print("Fitting...")
    svc.fit(x_train_count, y_train)
    print("-"*50)
    print("Predicting...")
    y_pred = svc.predict(x_to_predict)
    y_pred = encoder.inverse_transform(y_pred)
    print("-"*50)
    result[classifier_type] = y_pred

#Read data
train_dataset, test_dataset = read_data()

#Classifiers
classifier_types = ["Binary", "Multiclass"]

tfidt_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'([^,]+)',  ngram_range=(3,3))
tfidt_vect_ngram.fit(train_dataset["instructions"])

#svc = SVC(kernel="rbf", gamma="scale")
svc = MultinomialNB()

for classifier_type in classifier_types:
    #Split dataset
    encoder = preprocessing.LabelEncoder()
    x_train, y_train = split_dataset(classifier_type, encoder)
    x_train_count = tfidt_vect_ngram.transform(x_train)
    x_to_predict = tfidt_vect_ngram.transform(test_dataset["instructions"])
    classify(x_train_count, y_train, x_to_predict, svc, classifier_type, encoder)

print(result)