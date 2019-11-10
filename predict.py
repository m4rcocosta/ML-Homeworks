import json
import sys
from os.path import join
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
        instr = []
        mnemonics = []
        for instruction in instructions:
            instr.append(instruction)
            mnemonics.append(instruction.split(" ")[0])
        result["instructions"] = ",".join(instr)
        result["mnemonics"] = ",".join(mnemonics)
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
        instr = []
        mnemonics = []
        for instruction in instructions:
            instr.append(instruction)
            mnemonics.append(instruction.split(" ")[0])
        result["instructions"] = ",".join(instr)
        result["mnemonics"] = ",".join(mnemonics)
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

# Split dataset
def split_dataset(classifier_type, instruction_type, encoder):
    target = ""
    if classifier_type == "Optimization":
        target = "opt"
    else:
        target = "compiler"
    x_train, x_test, y_train, y_test = train_test_split(train_dataset[instruction_type], train_dataset[target], test_size = 0.20)
    y_train = encoder.fit_transform(y_train)
    return x_train, y_train

# Classify
def classify(x_train_count, y_train, x_to_predict, classifier, classifier_type, encoder):
    print("="*50)
    print("[Classifier type: " + classifier_type + "]")
    print("="*50)
    print("Fitting...")
    classifier.fit(x_train_count, y_train)
    print("-"*50)
    print("Predicting...")
    y_pred = classifier.predict(x_to_predict)
    y_pred = encoder.inverse_transform(y_pred)
    print("-"*50)
    result[classifier_type] = y_pred

# Instruction types
instruction_types = ["mnemonics", "instructions"]

# Classifiers
classifier_types = ["Optimization", "Compiler"]
classifiers = [LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto", max_iter=5000, n_jobs=-1), MultinomialNB(), DecisionTreeClassifier(random_state=0), SVC(kernel="rbf", gamma="scale"), KNeighborsClassifier(n_neighbors=3, n_jobs=-1), RandomForestClassifier(n_estimators=100, n_jobs=-1)]

# Vectorizers
vectorizers = [CountVectorizer(input="content", encoding="utf-8", tokenizer=lambda x: x.split(",")), TfidfVectorizer(analyzer="word", token_pattern=r"([^,]+)"), CountVectorizer(input="content", encoding="utf-8", tokenizer=lambda x: x.split(","),  ngram_range=(3,3)), TfidfVectorizer(analyzer="word", token_pattern=r"([^,]+)",  ngram_range=(3,3)), CountVectorizer(input="content", encoding="utf-8", tokenizer=lambda x: x.split(","),  ngram_range=(4,4)), TfidfVectorizer(analyzer="word", token_pattern=r"([^,]+)",  ngram_range=(4,4))]

# Main
if __name__ == "__main__":    
    if len(sys.argv) < 7:
        print("Bad usage! You must specify thie instruction types, the classifiers and the vectorizers for optimization and compiler classification.")
        exit(1)
    optimization_instruction_type = sys.argv[1]
    compiler_instruction_type = sys.argv[4]
    if optimization_instruction_type not in instruction_types or compiler_instruction_type not in instruction_types:
        print("Bad usage! Classifier you must insert the instruction type (mnemonics/instructions).")
        exit(1)
    optimization_classifier_index = sys.argv[2]
    compiler_classifier_index = sys.argv[5]
    if not optimization_classifier_index.isdigit() or not compiler_classifier_index.isdigit():
        print("Bad usage! Classifier must be an integer:")
        print("0 - Logistic Regression")
        print("1 - Multinomial NB")
        print("2 - Decision Tree")
        print("3 - SVC")
        print("4 - K-Neighbors")
        print("5 - Random Forest")
        exit(1)
    optimization_classifier_index = int(optimization_classifier_index)
    compiler_classifier_index = int(compiler_classifier_index)
    if optimization_classifier_index < 0 or optimization_classifier_index > 5 or compiler_classifier_index < 0 or compiler_classifier_index > 5:
        print("Bad usage! Classifier must be an integer:")
        print("0 - Logistic Regression")
        print("1 - Multinomial NB")
        print("2 - Decision Tree")
        print("3 - SVC")
        print("4 - K-Neighbors")
        print("5 - Random Forest")
        exit(1)
    optimization_vectorizer_index = sys.argv[3]
    compiler_vectorizer_index = sys.argv[6]
    if not optimization_vectorizer_index.isdigit() or not compiler_vectorizer_index.isdigit():
        print("Bad usage! Vectorizer must be an integer:")
        print("0 - Count Vectorizer")
        print("1 - Tfidf Vectorizer")
        print("2 - Count Vectorizer with ngram_range=(3,3)")
        print("3 - Tfidf Vectorizer with ngram_range=(3,3)")
        print("4 - Count Vectorizer with ngram_range=(4,4)")
        print("5 - Tfidf Vectorizer with ngram_range=(4,4)")
        exit(1)
    optimization_vectorizer_index = int(optimization_vectorizer_index)
    compiler_vectorizer_index = int(compiler_vectorizer_index)
    if optimization_vectorizer_index < 0 or optimization_vectorizer_index > 5 or compiler_vectorizer_index < 0 or compiler_vectorizer_index > 5:
        print("Bad usage! Vectorizer must be an integer:")
        print("0 - Count Vectorizer")
        print("1 - Tfidf Vectorizer")
        print("2 - Count Vectorizer with ngram_range=(3,3)")
        print("3 - Tfidf Vectorizer with ngram_range=(3,3)")
        print("4 - Count Vectorizer with ngram_range=(4,4)")
        print("5 - Tfidf Vectorizer with ngram_range=(4,4)")
        exit(1)
    optimization_classifier = classifiers[optimization_classifier_index]
    compiler_classifier = classifiers[compiler_classifier_index]
    optimization_vectorizer = vectorizers[optimization_vectorizer_index]
    compiler_vectorizer = vectorizers[compiler_vectorizer_index]

    # Read data
    train_dataset, test_dataset = read_data()
    print(train_dataset)
    print(test_dataset)
    print()

    # Encoder
    encoder = preprocessing.LabelEncoder()

    # Optimization classification
    print("Performing optimization classification...")
    x_train, y_train = split_dataset("Optimization", optimization_instruction_type, encoder)
    optimization_vectorizer.fit(train_dataset[optimization_instruction_type])
    x_train_count = optimization_vectorizer.transform(x_train)
    x_to_predict = optimization_vectorizer.transform(test_dataset[optimization_instruction_type])
    classify(x_train_count, y_train, x_to_predict, optimization_classifier, "Optimization", encoder)

    # Compiler classification
    print("Performing compiler classification...")
    x_train, y_train = split_dataset("Compiler", compiler_instruction_type, encoder)
    compiler_vectorizer.fit(train_dataset[compiler_instruction_type])
    x_train_count = compiler_vectorizer.transform(x_train)
    x_to_predict = compiler_vectorizer.transform(test_dataset[compiler_instruction_type])
    classify(x_train_count, y_train, x_to_predict, compiler_classifier, "Compiler", encoder)

    # Create csv file
    print("Writing csv file...")
    low_count = 0
    high_count = 0
    gcc_count = 0
    icc_count = 0
    clang_count = 0
    predictions_file = open("./predictions/predictions.csv", "w")
    count_file = open("./predictions/count.txt", "w")
    for i in range(len(result["Optimization"])):
        print(result["Compiler"][i] + "," + result["Optimization"][i], file = predictions_file)
        if result["Optimization"][i] == "L":
            low_count += 1
        else:
            high_count += 1
        if result["Compiler"][i] == "gcc":
            gcc_count += 1
        elif result["Compiler"][i] == "icc":
            icc_count += 1
        else:
            clang_count += 1
    print("Optimization classification result:", file = count_file)
    print("L: " + str(low_count) + ", H: " + str(high_count), file = count_file)
    print("Compiler classification result:", file = count_file)
    print("gcc: " + str(gcc_count) + ", icc: " + str(icc_count) + ", clang: " + str(clang_count), file = count_file)
    predictions_file.close()
    count_file.close()