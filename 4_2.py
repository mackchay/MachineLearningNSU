import torch
from torch import nn
import numpy as np
import math as m
import matplotlib.pyplot as plt
import scipy.stats as stats

epochs = 50000
nneuron_layers_num = 5

class MLP(nn.Module):
    def __init__(self, n, activation_func):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, n[0]),
            activation_func(),
            nn.Linear(n[0], n[1]),
            activation_func(),
            nn.Linear(n[1], n[2]),
            activation_func(),
            nn.Linear(n[2], 1),
        )

    def forward(self, x):
        return self.layers(x)


def fit(x_train, y_train, mlp, epoch, lr):
    torch.manual_seed(42)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=lr)

    xx_train = torch.from_numpy(x_train).type(torch.FloatTensor)
    xx_train = xx_train.reshape([len(x_train), 1])
    yy_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    for epoch in range(0, epoch + 1):

        mlp.train()
        optimizer.zero_grad()
        y_pred = mlp(xx_train).squeeze()
        loss = loss_function(y_pred, yy_train)
        loss.backward()
        optimizer.step()
        if (epoch % 100 == 0):
            print(f"Epoch: {epoch} | loss: {loss} ")


print("Мощность выборки:")
N = int(input())
print("Ошибка:")
eps0 = float(input())
print("Распределение ошибки: \n [1] равномерное; \n [2] нормальное;")
mode1 = int(input())
if mode1 == 2:
    print("Мат.ожидание:")
    mu = float(input())
    print("СКО:")
    sigma = float(input())
print("Функция: \n [1] a*x^3 + b*x^2 + c*x + d; \n [2] x*sin(2pi*x);")
mode2 = int(input())
if mode2 == 1:
    a = np.random.uniform(-3, 3, 4)


def genEps(mode):
    if mode == 1:
        return np.random.uniform(-eps0, eps0, N)
    else:
        return stats.truncnorm(-eps0, eps0, mu, sigma).rvs(N)


def f(x, mode):
    if mode == 1:
        return a[0] * x * x * x + a[1] * x * x + a[2] * x + a[3]
    else:
        return x * m.sin(2 * m.pi * x)


y = np.empty(N)
eps = genEps(mode1)

x = np.random.uniform(-1, 1, N)

for i in range(N):
    y[i] = f(x[i], mode2) + eps[i]

### реальная функция

plt.scatter(x, y, color='m')
xx = np.arange(-1, 1, 0.001)
if mode2 == 1:
    yy = a[0] * xx * xx * xx + a[1] * xx * xx + a[2] * xx + a[3]
    plt.plot(xx, yy, 'k', label='a*x^3+b*x^2+c*x+d')
else:
    yy = xx * np.sin(2 * m.pi * xx)
    plt.plot(xx, yy, 'k', label='x*sin(2*pi*x)')

##### из задания 1

print("M = ")
M = int(input())

left_side = np.zeros((M + 1, M + 1))
right_side = np.zeros(M + 1)
for k in range(M + 1):
    for i in range(M + 1):
        for j in range(N):
            left_side[k, i] += m.pow(x[j], i + k)
            if k == 0:
                right_side[i] += y[j] * m.pow(x[j], i)

sol = np.linalg.solve(left_side, right_side)
xx = np.arange(-1, 1, 0.001)
y_approach = np.zeros(2000)
for i in range(2000):
    for j in range(M + 1):
        y_approach[i] += sol[j] * m.pow(xx[i], j)

plt.plot(xx, y_approach, color='orange', label='approach')

##### многослойный перцептрон

mlp = MLP([nneuron_layers_num, nneuron_layers_num, nneuron_layers_num], nn.Tanh)
fit(x, y, mlp, epochs, 0.4)

mlp.eval()
x = torch.from_numpy(np.asarray(xx)).type(torch.FloatTensor)
x = x.reshape([len(xx), 1])
yy = mlp(x)

plt.plot(xx, yy.detach().numpy(), 'c', label='predicted')

######

print(f"M = {M}")
plt.legend(fontsize=10)
plt.ylim(-2, 2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
