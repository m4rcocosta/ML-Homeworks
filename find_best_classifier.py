import json
import sys
from os.path import join
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

# Train dataset path
train_dataset_path = "./dataset/train_dataset.jsonl"

# Read data
def read_data():
    print("Reading data ...")
    dataset = []
    with open(train_dataset_path, 'r') as json_file:
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

# Split dataset
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
    return x_train, x_test, y_train, y_test

# Classify
def classify(x_train_count, x_test_count, y_train, y_test, vectorizer, instruction_type, classifier_type):
    for classifier in classifiers:
        print("="*50, file = score_file)
        print("Classifier:",type(classifier).__name__,"[Vectorizer: " + vectorizer +"], [Instruction type: " + instruction_type +"], [Classifier type: " + classifier_type + "]", file = score_file)
        print("="*50, file = score_file)
        print("Fitting",type(classifier).__name__,"[Vectorizer: " + vectorizer +"], [Instruction type: " + instruction_type +"], [Classifier type: " + classifier_type + "]")
        classifier.fit(x_train_count, y_train)
        print("-"*50)
        print("Predicting",type(classifier).__name__,"[Vectorizer: " + vectorizer +"], [Instruction type: " + instruction_type +"], [Classifier type: " + classifier_type + "]")
        y_pred = classifier.predict(x_test_count)
        print("-"*50)
        print("Result:", file = score_file)
        print("-"*50, file = score_file)
        print("Confusion Matrix:", file = score_file)
        print(confusion_matrix(y_test, y_pred), file = score_file)
        print("-"*50, file = score_file)
        print("Classification Report:", file = score_file)
        print(classification_report(y_test, y_pred), file = score_file)
        print("-"*50, file = score_file)
        print("Accuracy:", file = score_file)
        print(accuracy_score(y_test, y_pred), file = score_file)
        print("-"*50, file = score_file)
        print("Precision:", file = score_file)
        print(precision_score(y_test, y_pred, average="macro"), file = score_file)
        print("-"*50, file = score_file)
        print("Recall:", file = score_file)
        print(recall_score(y_test, y_pred, average="macro"), file = score_file)
        print("-"*50, file = score_file)
        print("Performing Cross Validation Score",type(classifier).__name__,"[Vectorizer: " + vectorizer +"], [Instruction type: " + instruction_type +"], [Classifier type: " + classifier_type + "]")
        print("Cross Validation Score:", file = score_file)
        print(cross_val_score(classifier, x_test_count, y_test, cv = 10, n_jobs = -1), file = score_file)
        print("-"*50)
        print("-"*50, file = score_file)
        print("F1 Score:", file = score_file)
        print(f1_score(y_test, y_pred, average="macro"), file = score_file)
        print("-"*50, file = score_file)


# Classifiers
classifier_types = ["Binary", "Multiclass"]
classifiers = [LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto", max_iter=5000, n_jobs=-1), MultinomialNB(), DecisionTreeClassifier(random_state=0), SVC(kernel="rbf", gamma="scale"), KNeighborsClassifier(n_neighbors=3, n_jobs=-1), RandomForestClassifier(n_estimators=100, n_jobs=-1)]

# Vectorizers
vectorizers = [CountVectorizer(input="content", encoding="utf-8", tokenizer=lambda x: x.split(",")), TfidfVectorizer(analyzer="word", token_pattern=r"([^,]+)"), CountVectorizer(input="content", encoding="utf-8", tokenizer=lambda x: x.split(","),  ngram_range=(3,3)), TfidfVectorizer(analyzer="word", token_pattern=r"([^,]+)",  ngram_range=(3,3)), CountVectorizer(input="content", encoding="utf-8", tokenizer=lambda x: x.split(","),  ngram_range=(4,4)), TfidfVectorizer(analyzer="word", token_pattern=r"([^,]+)",  ngram_range=(4,4))]

# Instruction types
instruction_types = ["mnemonics", "instructions"]

# Main
if len(sys.argv) < 3:
    print("Bad usage! You must specify the classifier type (Binary/Multiclass) and the instruction type (mnemonics/instructions).")
    exit(1)
classifier_type = sys.argv[1]
instruction_type = sys.argv[2]
if classifier_type not in classifier_types:
    print("Bad usage! Classifier type must be 'Binary' or 'Multiclass'.")
    exit(1)
if instruction_type not in instruction_types:
    print("Bad usage! Instruction type must be 'mnemonics' or 'instructions'.")
    exit(1)

# Read data
data = read_data()
print(data)
print()

score_file = open("./scores/" + classifier_type + "_" + instruction_type + "_scores.txt", "w")

# Split dataset
x_train, x_test, y_train, y_test = split_dataset(classifier_type, instruction_type)
for vectorizer in vectorizers:
    vectorizer.fit(data[instruction_type])
    x_train_count = vectorizer.transform(x_train)
    x_test_count = vectorizer.transform(x_test)
    name = type(vectorizer).__name__
    if vectorizer == vectorizers[2] or vectorizer == vectorizers[3]:
        name += " Ngram Range 3"
    elif vectorizer == vectorizers[4] or vectorizer == vectorizers[5]:
        name += " Ngram Range 4"
    classify(x_train_count, x_test_count, y_train, y_test, name, instruction_type, classifier_type)

score_file.close()
