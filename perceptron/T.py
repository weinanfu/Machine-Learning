from random import seed
from random import randrange
from csv import reader
import numpy as np


# Load a CSV file
def load_csv(filename):
    unknown = '?'
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for index, row in enumerate(csv_reader):
            if unknown in list(row) or index == 0:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert third column
def int_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])

def int_column_to_floatone(dataset, column):
    for row in dataset:
        row[column] = float(row[column]/10000)

def int_column_to_floattwo(dataset, column):
    for row in dataset:
        row[column] = float(row[column]/1000)

# Convert string column to integer
def str_column_to_index(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores



seed(1)
# load and prepare data
filename = 'adult.csv'
dataset = load_csv(filename)
matrix = [0,2,4,10,11,12]
for i in range(len(dataset[0])):
    if i in matrix:
        str_column_to_float(dataset, i)
    else:
        str_column_to_index(dataset, i)
        int_column_to_float(dataset, i)
int_column_to_floatone(dataset, 2)
int_column_to_floattwo(dataset, 11)
int_column_to_floattwo(dataset, 10)
m = np.array(dataset)
dataMat = m[:,0:len(m[0])-1]
labelMat = m[:,len(m[0])-1]

print(labelMat)
