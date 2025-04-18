import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.read_csv('docs/iris.csv')
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_normalized = (X - X_mean) / X_std

def one_hot_encode(y_labels):
    classes = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
    y_encoded = np.zeros((len(y_labels), 3))
    for i, label in enumerate(y_labels):
        y_encoded[i, classes[label]] = 1
    return y_encoded

y_labels = df['variety'].values
y_onehot = one_hot_encode(y_labels)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X_normalized = X_normalized[indices]
y_onehot = y_onehot[indices]

train_size = int(0.8 * len(X))
X_train, y_train = X_normalized[:train_size], y_onehot[:train_size]
X_test, y_test = X_normalized[train_size:], y_onehot[train_size:]

n_neurons = int(input("Perceptrons na camada oculta: "))
learning_rate = float(input("Taxa de aprendizado: "))
epochs = int(input("Quantidade de epochs: "))
batch_size = int(input("Tamanho do batch: "))

class Perceptron:
    def __init__(self, n_input):
        self.weights = np.random.randn(n_input) * np.sqrt(2 / n_input)
        self.bias = 0

    def forward(self, X):
        self.z = np.dot(X, self.weights) + self.bias
        return self.z

class Layer:
    def __init__(self, n_inputs, n_neurons, activation):
        self.activation = activation
        self.neurons = [Perceptron(n_inputs) for _ in range(n_neurons)]

    def forward(self, X):
        self.last_input = X
        self.z_outputs = np.array([neuron.forward(X) for neuron in self.neurons]).T
        if self.activation == "relu":
            self.a_outputs = np.maximum(0, self.z_outputs)
        elif self.activation == "softmax":
            self.a_outputs = self.softmax(self.z_outputs)
        return self.a_outputs

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def backward(self, dA, y=None, learning_rate=0.01):
        m = self.last_input.shape[0]

        if self.activation == "softmax":
            dZ = self.a_outputs - y
        elif self.activation == "relu":
            dZ = dA * (self.z_outputs > 0)

        for i, neuron in enumerate(self.neurons):
            dW = np.dot(self.last_input.T, dZ[:, i]) / m
            db = np.sum(dZ[:, i]) / m
            neuron.weights -= learning_rate * dW
            neuron.bias -= learning_rate * db

        return np.dot(dZ, np.array([neuron.weights for neuron in self.neurons]))

hidden_layer = Layer(n_inputs=4, n_neurons=n_neurons, activation="relu")
output_layer = Layer(n_inputs=n_neurons, n_neurons=3, activation="softmax")

for epoch in range(epochs):

    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        hidden_output = hidden_layer.forward(X_batch)
        output = output_layer.forward(hidden_output)

        loss = -np.sum(y_batch * np.log(output + 1e-9)) / X_batch.shape[0]

        d_hidden = output_layer.backward(None, y_batch, learning_rate)
        hidden_layer.backward(d_hidden, learning_rate=learning_rate)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


hidden_test = hidden_layer.forward(X_test)
test_output = output_layer.forward(hidden_test)
predictions = np.argmax(test_output, axis=1)
true_labels = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == true_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")