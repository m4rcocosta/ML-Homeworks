import json
import sys
from os.path import join
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.svm import SVC

path = "./dataset/train_dataset.jsonl"
path2 = "./dataset/test_dataset_blind.jsonl"

# Read data
def read_data():
    print("Reading data ...")
    dataset = []
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        instructions = result["instructions"]
        mnemonics = []
        for instruction in instructions:
            mnemonics.append(instruction.split(" ")[0])
        result["instructions"] = ",".join(mnemonics)
        dataset.append(result)

    data = pd.DataFrame(dataset)
    print("Data read successfully!")
    print("-"*50)
    print(data.shape)
    print("-"*50)
    return data

#split dataset
def split_dataset(classifier_type):
    target = ""
    if classifier_type == "Binary":
        target = "opt"
    else:
        target = "compiler"
    x_train, x_test, y_train, y_test = train_test_split(data["instructions"], data[target], test_size = 0.20)
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)
    sys.stdout.write("x_train = ")
    print(x_train.shape)
    sys.stdout.write("y_train = ")
    print(y_train.shape)
    print("-"*50)
    sys.stdout.write("x_test = ")
    print(x_test.shape)
    sys.stdout.write("y_test = ")
    print(y_test.shape)
    print("-"*50)
    return x_train, x_test, y_train, y_test

#classify
def classify(x_train_count, x_test_count, vectorizer, classifier_type):
    print("="*50)
    print("[Classifier type: " + classifier_type + "]")
    print("="*50)
    print("Fitting...")
    svc.fit(x_train_count, y_train)
    print("-"*50)
    print("Predicting...")
    y_pred = svc.predict(x_test_count)
    print("-"*50)

#Read data
data = read_data()

#Classifiers
classifier_types = ["Binary", "Multiclass"]

tfidt_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'([^,]+)',  ngram_range=(3,3))
tfidt_vect_ngram.fit(data["instructions"])

svc = SVC(kernel="rbf", gamma="scale")

for classifier_type in classifier_types:
    #Split dataset
    x_train, x_test, y_train, y_test = split_dataset(classifier_type)
    x_train_count = tfidt_vect_ngram.transform(x_train)
    x_test_count = tfidt_vect_ngram.transform(x_test)
    classify(x_train_count, x_test_count, classifier_type)