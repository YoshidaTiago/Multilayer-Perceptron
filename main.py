import pandas as pd
import numpy as np

df = pd.read_csv('Multilayer-Perceptron\\docs\\iris.csv')

X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values

class Perceptron():
    def __init__(self, input_size, learning_rate):
        self.weights = np.zeros(input_size)
        self.learning_rate = learning_rate
        self.bias = 0