import pandas as pd
from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm, datasets, metrics
import numpy as np
from define_class import DefineClass
from sklearn.model_selection import GridSearchCV
import joblib
import time
from sklearn.linear_model import LogisticRegression

class Logistic_Regression(DefineClass):
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
        # param_grid = {
        #     'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        #     'C': [0.01, 0.1, 1, 10, 100],
        #     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        #     'max_iter': [1000, 2000, 5000, 10000]
        # }
        # grid_search = GridSearchCV(LogisticRegression(random_state=0), param_grid, cv=5, scoring='accuracy')
        # start_time = time.time()
        # grid_search.fit(self.X_train, self.y_train)
        # end_time = time.time()
        # print(f'Best parameters found: {grid_search.best_params_}')
        # print(f'Best accuracy: {grid_search.best_score_}')
        #self.clf = grid_search.best_estimator_
        
        self.clf = LogisticRegression(random_state=0, max_iter=10000)
        start_time = time.time()
        self.clf.fit(self.X_train, self.y_train)
        end_time = time.time()
        print(f'logistic: Time taken for training: {end_time - start_time:.2f} seconds')

    def predict(self):
        self.y_pred = self.clf.predict(self.X_test)
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        return self.accuracy

    def evaluate(self):
        print(f'Accuracy (Logistic regression): {self.accuracy}')
        print(metrics.classification_report(self.y_test, self.y_pred, zero_division=1))
        print(metrics.confusion_matrix(self.y_test, self.y_pred))

    def plot(self):
        self.disp = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(self.y_test, self.y_pred))
        self.disp.plot()
        plt.show()
    
    def save_model(self, model_file):
        import joblib
        joblib.dump(self.clf, model_file)