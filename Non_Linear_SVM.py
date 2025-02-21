import pandas as pd
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm, metrics
import numpy as np
from define_class import DefineClass
from sklearn.model_selection import GridSearchCV
import joblib
import time

class Non_Linear_SVM(DefineClass):
    def __init__(self, train_file, test_file, train_size, test_size):
        self.train_file = train_file
        self.test_file = test_file

        self.train_size = train_size
        self.test_size = test_size
        
        self.train_data = pd.read_csv(self.train_file)
        self.test_data = pd.read_csv(self.test_file)
        
        self.train_sample = self.train_data.sample(n=self.train_size, random_state=42)
        self.test_sample = self.test_data.sample(n=self.test_size, random_state=42)
        
        self.X_train = self.train_sample.iloc[:, 1:].values
        self.y_train = self.train_sample.iloc[:, 0].values
        
        self.X_test = self.test_sample.iloc[:, 1:].values
        self.y_test = self.test_sample.iloc[:, 0].values

        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

    def train(self):
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'max_iter':[15000, 20000],
            'degree': [2, 3, 4]
        }
        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy')
        start_time = time.time()
        grid_search.fit(self.X_train, self.y_train)
        end_time = time.time()
        self.clf = grid_search.best_estimator_
        print(f'Best parameters found: {grid_search.best_params_}')
        print(f'Non-Linear SVM: Time taken for grid search: {end_time - start_time:.2f} seconds')


    def predict(self):
        self.y_pred = self.clf.predict(self.X_test)
        self.accuracy_non_linear = metrics.accuracy_score(self.y_test, self.y_pred)
        return self.accuracy_non_linear

    def evaluate(self):
        print(f'Accuracy (Non-Linear SVM): {self.accuracy_non_linear}')
        print(metrics.classification_report(self.y_test, self.y_pred, zero_division=1))
        print(metrics.confusion_matrix(self.y_test, self.y_pred))

    def plot(self):
        self.disp = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(self.y_test, self.y_pred))
        self.disp.plot()
        plt.show()
    
    def save_model(self, model_file):
        joblib.dump(self.clf, model_file)
        print(f"Model saved to {model_file}")

    def build(self) -> DefineClass:
        return DefineClass()

# Best parameters found: {'C': 10, 'degree': 2, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': 1000}
# Accuracy (Non-Linear SVM): 0.99
#               precision    recall  f1-score   support

#            0       0.99      1.00      0.99        87
#            1       1.00      1.00      1.00       104
#            2       0.99      0.98      0.99       114
#            3       0.97      1.00      0.99       102
#            4       0.98      0.99      0.98        94
#            5       1.00      1.00      1.00       101
#            6       0.99      0.96      0.97       112
#            7       0.98      1.00      0.99        94
#            8       1.00      1.00      1.00        84
#            9       1.00      0.98      0.99       108

#     accuracy                           0.99      1000
#    macro avg       0.99      0.99      0.99      1000
# weighted avg       0.99      0.99      0.99      1000

# [[ 87   0   0   0   0   0   0   0   0   0]
#  [  0 104   0   0   0   0   0   0   0   0]
#  [  0   0 112   0   1   0   1   0   0   0]
#  [  0   0   0 102   0   0   0   0   0   0]
#  [  0   0   0   1  93   0   0   0   0   0]
#  [  0   0   0   0   0 101   0   0   0   0]
#  [  1   0   1   2   1   0 107   0   0   0]
#  [  0   0   0   0   0   0   0  94   0   0]
#  [  0   0   0   0   0   0   0   0  84   0]
#  [  0   0   0   0   0   0   0   2   0 106]]