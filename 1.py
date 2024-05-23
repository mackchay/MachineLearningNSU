import math
import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import solve, lstsq

a, b, c, d = np.random.uniform(-3, 3, size=4)
N = 15
m = 20
epsilon = 0.5
x = np.random.uniform(-1, 1, N)
x.sort()


class NonExistingChoice(Exception):
    pass


def func_1(arg):
    return a * arg ** 3 + b * arg ** 2 + c * arg + d


def func_2(arg):
    return arg * math.sin(2 * math.pi * arg)


epsilon_distribution_choice = input("enter epsilon distribution: ")
if epsilon_distribution_choice == 'a':
    print('epsilon distributed uniformly\n')
    epsilons = np.random.uniform(-epsilon, epsilon, size=N)
elif epsilon_distribution_choice == 'b':
    print('epsilon distributed normally\n')
    epsilons = np.random.normal(loc=0, scale=epsilon, size=N)
else:
    raise NonExistingChoice()

print("choose func: a or b: ")
fun_type = input()
if fun_type == 'a':
    func = func_1
elif fun_type == 'b':
    func = func_2
else:
    raise NonExistingChoice()

x_values = np.linspace(-1, 1, 100)
y_values = list(map(lambda x_i: func(x_i), x_values))
y_plus_e = list(map(lambda x_i, epsilon_i: func(x_i) + epsilon_i, x, epsilons))

print('x = ', x)
print('y =', y_plus_e)

A = []
B = []

len_x = len(x)
for i in range(m):
    B.append(sum(x[k] ** i * y_plus_e[k] for k in range(len_x)))
    A.append(list(sum(x[k] ** (i + j) for k in range(len_x)) for j in range(N)))


w = lstsq(A, B)[0]

print("w = ", w)


def polynom(new_xs):
    return sum(new_xs ** i * w[i] for i in range(N))


xs = np.linspace(-1, 1, 1000)
plt.scatter(x, y_plus_e)
plt.plot(x_values, y_values, 'b')
plt.plot(xs, polynom(xs), 'r')
plt.ylim(-5, 5)
plt.xlabel('x')
plt.ylabel('y')
plt.show()





