import abc

class DefineClass(abc.ABC):
    @abc.abstractmethod
    def train(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    @abc.abstractmethod
    def evaluate(self):
        pass
