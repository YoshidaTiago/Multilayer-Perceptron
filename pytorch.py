import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv('docs/iris.csv')
X = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
y_labels = df['variety'].values

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

label_map = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
y = np.array([label_map[label] for label in y_labels])

indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

n_neurons = int(input("Perceptrons na camada oculta: "))
learning_rate = float(input("Taxa de aprendizado: "))
epochs = int(input("Quantidade de epochs: "))
batch_size = int(input("Tamanho do batch: "))

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

model = MLP(input_size=4, hidden_size=n_neurons, output_size=3)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    permutation = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean().item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
