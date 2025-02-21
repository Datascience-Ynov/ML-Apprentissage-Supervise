from myMLP import myMLP
from Linear_SVM import Linear_SVM
from Linear_SVM_Optimized import Linear_SVM_Optimized
from Non_Linear_SVM import Non_Linear_SVM
from XGBoost import XGBoost_classifier
from Random_forest import Random_Forest
from Logistic_Regression import Logistic_Regression
from KNN import KNN

class main:
    def __init__(self):
        self.train_file = 'fashion-mnist_train.csv'
        self.test_file = 'fashion-mnist_test.csv'
        self.accuracies = {}
        self.train_size = 2000
        self.test_size = 1000

    def MLP_test(self):
        mlp = myMLP(self.train_file, self.test_file, self.train_size, self.test_size)
        mlp.train()
        self.accuracies['MLP'] = mlp.predict()
        mlp.evaluate()

    def Linear_SVM_test(self):
        svm = Linear_SVM(self.train_file, self.test_file, self.train_size, self.test_size)
        svm.train()
        self.accuracies['SVM'] = svm.predict()
        svm.evaluate()
    
    def Linear_SVM_Optimized_test(self):
        svm = Linear_SVM_Optimized(self.train_file, self.test_file, self.train_size, self.test_size)
        svm.train()
        self.accuracies['SVM_OP'] = svm.predict()
        svm.evaluate()
    
    def Non_Linear_SVM_test(self):
        svm = Non_Linear_SVM(self.train_file, self.test_file, self.train_size, self.test_size)
        svm.train()
        accuracy_non_linear = svm.predict()
        self.accuracies['SVM_NL'] = accuracy_non_linear
        svm.evaluate()
    
    def XGBoost_test(self):
        xgb = XGBoost_classifier(self.train_file, self.test_file, self.train_size, self.test_size)
        xgb.train()
        self.accuracies['XGBoost'] = xgb.predict()
        xgb.evaluate()
    
    def Random_Forest_test(self):
        rf = Random_Forest(self.train_file, self.test_file, self.train_size, self.test_size)
        rf.train()
        self.accuracies['Random_Forest'] = rf.predict()
        rf.evaluate()
    
    def Logistic_Regression_test(self):
        lr = Logistic_Regression(self.train_file, self.test_file, self.train_size, self.test_size)
        lr.train()
        self.accuracies['Logistic_Regression'] = lr.predict()
        lr.evaluate()
    
    def KNN_test(self):
        knn = KNN(self.train_file, self.test_file, self.train_size, self.test_size)
        knn.train()
        self.accuracies['KNN'] = knn.predict()
        knn.evaluate()
    
    def run_all_tests(self):
        self.MLP_test()
        self.Linear_SVM_test()
        self.Linear_SVM_Optimized_test()
        self.Non_Linear_SVM_test()
        self.XGBoost_test()
        self.Random_Forest_test()
        self.Logistic_Regression_test()
        self.KNN_test()

    def compare_accuracies(self):
        sorted_accuracies = sorted(self.accuracies.items(), key=lambda item: item[1], reverse=True)
        print("Model Rankings based on Accuracy:")
        lst = []
        for model, accuracy in sorted_accuracies:
            lst.append(model)
            print(f"{model}: {accuracy}")
        return lst[:2]
    
    def train_best_models_on_full_data(self):
        self.run_all_tests()
        best_models = self.compare_accuracies()
        self.train_size = 60000
        self.test_size = 10000
        
        for model_name in best_models:
            if model_name == 'MLP':
                model = myMLP(self.train_file, self.test_file, self.train_size, self.test_size)
            elif model_name == 'SVM':
                model = Linear_SVM(self.train_file, self.test_file, self.train_size, self.test_size)
            elif model_name == 'SVM_OP':
                model = Linear_SVM_Optimized(self.train_file, self.test_file, self.train_size, self.test_size)
            elif model_name == 'SVM_NL':
                model = Non_Linear_SVM(self.train_file, self.test_file, self.train_size, self.test_size)
            elif model_name == 'XGBoost':
                model = XGBoost_classifier(self.train_file, self.test_file, self.train_size, self.test_size)
            elif model_name == 'Random_Forest':
                model = Random_Forest(self.train_file, self.test_file, self.train_size, self.test_size)
            elif model_name == 'Logistic_Regression':
                model = Logistic_Regression(self.train_file, self.test_file, self.train_size, self.test_size)
            elif model_name == 'KNN':
                model = KNN(self.train_file, self.test_file, self.train_size, self.test_size)
            
            model.train()
            accuracy = model.predict()
            print(f'{model_name} ré-entrainé avec une exactitude de: {accuracy}')


if __name__ == "__main__":
    main_instance = main()
    main_instance.train_best_models_on_full_data()