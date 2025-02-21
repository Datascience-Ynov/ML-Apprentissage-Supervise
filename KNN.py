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
from sklearn.neighbors import KNeighborsClassifier

class KNN(DefineClass):
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
        #     'n_neighbors': [3, 5, 7, 10, 15],
        #     'weights': ['uniform', 'distance'],
        #     'leaf_size': [20, 30, 40],
        #     'p': [1, 2]
        # }
        # grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
        self.clf = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto', 
                    leaf_size=30)
        start_time = time.time()
        #grid_search.fit(self.X_train, self.y_train)
        self.clf.fit(self.X_train, self.y_train)
        end_time = time.time()
        #self.clf = grid_search.best_estimator_
        #print(f'Best parameters found: {grid_search.best_params_}')
        print(f'KNN: Time taken for training: {end_time - start_time:.2f} seconds')

    def predict(self):
        self.y_pred = self.clf.predict(self.X_test)
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)
        return self.accuracy

    def evaluate(self):
        print(f'Accuracy (KNN): {self.accuracy}')
        print(metrics.classification_report(self.y_test, self.y_pred, zero_division=1))
        print(metrics.confusion_matrix(self.y_test, self.y_pred))

    def plot(self):
        self.disp = metrics.ConfusionMatrixDisplay(confusion_matrix=metrics.confusion_matrix(self.y_test, self.y_pred))
        self.disp.plot()
        plt.show()
    
    def save_model(self, model_file):
        import joblib
        joblib.dump(self.clf, model_file)
        print(f"Model saved to {model_file}")

    def build(self) -> DefineClass:
        return DefineClass()