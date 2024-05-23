from keras.datasets import mnist
import numpy as np
import torch
from torch import nn

vector_len = 28 * 28
N = 1000
num_classes = 10


class MLP(nn.Module):
    def __init__(self, n, activation_func):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vector_len, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='float')[y]


def fit(x_train, y_train, x_test, y_test, mlp, epoch, lr):
    torch.manual_seed(42)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=lr)

    xx_train = torch.from_numpy(np.asarray(x_train)).type(torch.FloatTensor)
    yy_train = torch.from_numpy(np.asarray(y_train)).type(torch.FloatTensor)
    xx_test = torch.from_numpy(np.asarray(x_test)).type(torch.FloatTensor)
    yy_test = torch.from_numpy(np.asarray(y_test)).type(torch.FloatTensor)
    for epoch in range(0, epoch + 1):

        mlp.train()
        optimizer.zero_grad()
        y_pred = mlp(xx_train).squeeze()
        loss = loss_function(y_pred, yy_train)
        acc = accuracy_function(yy_train, y_pred)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            mlp.eval()
            with torch.inference_mode():
                test_pred = mlp(xx_test).squeeze()
                test_loss = loss_function(test_pred, yy_test)
                test_acc = accuracy_function(yy_test, test_pred)
            print(
                f"Epoch: {epoch} | loss: {loss} | accuracy: {acc}% | test_loss: {test_loss} | test_accuracy: {test_acc}")


def normal(y_pred):
    pred = y_pred.detach().numpy()
    y = []
    for i in range(len(pred)):
        num = np.argmax(pred[i])
        y.append(num)
    return y


def accuracy_function(y_true, y_pred):
    predicted_norm = torch.from_numpy(np.asarray(normal(y_pred))).type(torch.FloatTensor)
    true_norm = torch.from_numpy(np.asarray(normal(y_true))).type(torch.FloatTensor)
    correct = torch.eq(true_norm, predicted_norm).sum().item()
    #   print(correct)
    acc = (correct / len(y_pred)) * 100
    return acc


def predict(mlp, x_test, y_test):
    mlp.eval()
    xx_test = torch.from_numpy(np.asarray(x_test)).type(torch.FloatTensor)
    y_pred = mlp(xx_test).squeeze()
    return y_pred.detach().numpy()


def ensemble():
    epochs = 50
    perceptron_num = 5
    (x, y), (x1, y1) = mnist.load_data()
    res = np.zeros((100, num_classes))
    problem = x1[:100]
    problem = problem.reshape(problem.shape[0], vector_len)
    problem = problem.astype('float32')
    problem = problem / 255
    answer = to_categorical(y1[:100], num_classes)
    for i in range(perceptron_num):
        mlp = MLP([1], nn.ReLU)
        train_x = x[i * N:(i + 1) * N]
        test_x = x1[i * 100:(i + 1) * 100]
        train_x = train_x.reshape(train_x.shape[0], vector_len)
        test_x = test_x.reshape(test_x.shape[0], vector_len)
        train_x = train_x.astype('float32')
        test_x = test_x.astype('float32')
        train_x = train_x / 255
        test_x = test_x / 255
        train_y = to_categorical(y[i * N:(i + 1) * N], num_classes)
        test_y = to_categorical(y1[i * 100:(i + 1) * 100], num_classes)
        print(f"perceptron: {i + 1}")
        fit(train_x, train_y, test_x, test_y, mlp, epochs, 0.5)
        res += predict(mlp, problem, answer)

    res = torch.from_numpy(np.asarray(res)).type(torch.FloatTensor)
    answer = torch.from_numpy(np.asarray(answer)).type(torch.FloatTensor)
    acc = accuracy_function(answer, res)
    print(f"ensemble accuracy: {acc}%")


ensemble()
# print(test_y, y_pred)
# print(weights)
