import json
import sys
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

path = "./train_dataset.jsonl"
result_file = open("result.txt", "a")

# Read data
def read_data():
    print("Reading data ...")
    dataset = []
    with open(path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        result = json.loads(json_str)
        instructions = result["instructions"]
        instr = []
        mnemonics = []
        for instruction in instructions:
            instr.append(instruction)
            mnemonics.append(instruction.split(" ")[0])
        result["instructions"] = ",".join(instr)
        result["mnemonics"] = ",".join(mnemonics)
        dataset.append(result)

    data = pd.DataFrame(dataset)
    print("Data read successfully!")
    print("-"*50)
    print(data.shape)
    print("-"*50)
    return data

#split dataset
def split_dataset(classifier_type, instruction_type):
    target = ""
    if classifier_type == "Binary":
        target = "opt"
    else:
        target = "compiler"
    x_train, x_test, y_train, y_test = train_test_split(data[instruction_type], data[target], test_size = 0.20)
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
def classify(x_train_count, x_test_count, vectorizer, instruction_type, classifier_type):
    for classifier in classifiers:
        print("="*50, file = result_file)
        print("Classifier:",type(classifier).__name__,"[Vectorizer: " + vectorizer +"], [Instruction type: " + instruction_type +"], [Classifier type: " + classifier_type + "]", file = result_file)
        print("="*50, file = result_file)
        print("Fitting",type(classifier).__name__,"[Vectorizer: " + vectorizer +"], [Instruction type: " + instruction_type +"], [Classifier type: " + classifier_type + "]")
        classifier.fit(x_train_count, y_train)
        print("-"*50)
        print("Predicting",type(classifier).__name__,"[Vectorizer: " + vectorizer +"], [Instruction type: " + instruction_type +"], [Classifier type: " + classifier_type + "]")
        y_pred = classifier.predict(x_test_count)
        print("-"*50)
        print("Result:", file = result_file)
        print("-"*50, file = result_file)
        print("Confusion Matrix:", file = result_file)
        print(confusion_matrix(y_test, y_pred), file = result_file)
        print("-"*50, file = result_file)
        print("Classification Report:", file = result_file)
        print(classification_report(y_test, y_pred), file = result_file)
        print("-"*50, file = result_file)
        print("Accuracy:", file = result_file)
        print(accuracy_score(y_test, y_pred), file = result_file)
        print("-"*50, file = result_file)
        print("Precision:", file = result_file)
        print(precision_score(y_test, y_pred, average="macro"), file = result_file)
        print("-"*50, file = result_file)
        print("Recall:", file = result_file)
        print(recall_score(y_test, y_pred, average="macro"), file = result_file)
        print("-"*50, file = result_file)
        print("Cross Validation Score:", file = result_file)
        print(cross_val_score(classifier, x_test_count, y_test, cv=10), file = result_file)
        print("-"*50, file = result_file)
        print("F1 Score:", file = result_file)
        print(f1_score(y_test, y_pred, average="macro"), file = result_file)
        print("-"*50, file = result_file)

#Read data
data = read_data()

#Classifiers
classifier_types = ["Binary", "Multiclass"]
classifiers = [LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=5000), MultinomialNB(), SVC(kernel="rbf", gamma="scale")]

#Vectorizers
count_vect = CountVectorizer(input='content', encoding='utf-8', tokenizer=lambda x: x.split(','))
tfidt_vect = TfidfVectorizer(analyzer='word', token_pattern=r'([^,]+)')
tfidt_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'([^,]+)',  ngram_range=(3,3))
vectorizers = [count_vect, tfidt_vect, tfidt_vect_ngram]

instruction_types = ["mnemonics", "instructions"]


for classifier_type in classifier_types:
    for instruction_type in instruction_types:
        #Split dataset
        x_train, x_test, y_train, y_test = split_dataset(classifier_type, instruction_type)
        for vectorizer in vectorizers:
            vectorizer.fit(data[instruction_type])
            x_train_count = vectorizer.transform(x_train)
            x_test_count = vectorizer.transform(x_test)
            name = type(vectorizer).__name__
            if vectorizer == tfidt_vect_ngram:
                name += " Ngmam Range"
            classify(x_train_count, x_test_count, name, instruction_type, classifier_type)

result_file.close()
