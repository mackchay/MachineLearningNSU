from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import random
import time

from sklearn import metrics


class NonExistingChoice(Exception):
    pass


def generate_noise(eps0):
    return np.random.uniform(-eps0, eps0)


def generate_gaussian(number_of_samples, mu, sigma, error_scale):
    x_train = np.zeros((number_of_samples, 2), dtype=float)
    y_train = np.zeros(number_of_samples, dtype=int)

    for i in range(number_of_samples):

        for j in range(2):
            x_train[i][j] = np.random.normal(loc=mu[j], scale=sigma[j], size=None)

        y_train[i] = 1 if (sum(x_train[i]) > 0) else 0

        for j in range(2):
            x_train[i] += generate_noise(error_scale)

    return x_train, y_train


def gaussian(number_of_samples, mu, sigma, error_scale):
    x_train_p, y_train_p = generate_gaussian(int(number_of_samples / 2), mu, sigma, error_scale)
    x_train_n, y_train_n = generate_gaussian(int(number_of_samples / 2), -1 * mu, sigma, error_scale)
    return np.concatenate((x_train_p, x_train_n)), np.concatenate((y_train_p, y_train_n))


def circle(number_of_samples, radius, eps0):
    x_train = np.zeros((number_of_samples, 2), dtype=float)
    y_train = np.zeros(number_of_samples, dtype=int)

    n = int(number_of_samples / 2)

    for i in range(n):
        r = np.random.uniform(0, radius * 0.5)
        angle = np.random.uniform(0, 2 * np.pi)
        x_train[i] = [r * np.sin(angle) + generate_noise(eps0), r * np.cos(angle) + generate_noise(eps0)]
        y_train[i] = 0

    for i in range(n):
        R = np.random.uniform(radius * 0.7, radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x_train[n + i] = [R * np.sin(angle) + generate_noise(eps0), R * np.cos(angle) + generate_noise(eps0)]
        y_train[i] = 1

    return x_train, y_train


def xor_data(number_of_samples, start, stop, eps0):
    x_train = np.zeros((number_of_samples, 2), dtype=float)
    y_train = np.zeros(number_of_samples, dtype=int)
    for i in range(number_of_samples):
        x_train[i] = [np.random.uniform(start, stop), np.random.uniform(start, stop)]

        y_train[i] = 1 if (np.prod(np.array(x_train[i])) >= 0) else 0

        for j in range(2): x_train[i][j] += generate_noise(eps0)
    return x_train, y_train


def generate_spiral(number_of_samples, deltaT, label, eps0):
    x_train = np.zeros((number_of_samples, 2), dtype=float)
    y_train = np.zeros(number_of_samples, dtype=int)
    for i in range(number_of_samples):
        r = i / number_of_samples * 5
        t = 1.75 * i / number_of_samples * 2 * np.pi + deltaT
        x_train[i] = [r * np.sin(t) + generate_noise(eps0), r * np.cos(t) + generate_noise(eps0)]
        y_train[i] = label
    return x_train, y_train


def spiral_data(number_of_samples, eps0):
    x_train_p, y_train_p = generate_spiral(int(number_of_samples / 2), 0, 0, eps0)
    x_train_n, y_train_n = generate_spiral(int(number_of_samples / 2), np.pi, 1, eps0)
    return np.concatenate((x_train_p, x_train_n)), np.concatenate((y_train_p, y_train_n))


def create_axe(axe, x, y):
    axe.scatter(x[:, 0], x[:, 1], c=y, cmap='viridis')
    axe.set_xlabel("X_1-axis")
    axe.set_ylabel("X_2-axis")


def step_function(x):
    return np.heaviside(x, 1)


def f(weights, x):
    return (weights[0] + weights[1:-1] @ x) / (-weights[-1])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def algorithm_studying(x_train, y_train, coeff, epochs):
    w = np.zeros(shape=len(x_train[0]))

    for count in range(epochs):
        for i in range(len(x_train)):
            res = x_train[i] @ w

            if res > 0 and y_train[i] == 0:
                w -= coeff * x_train[i]
            elif res <= 0 and y_train[i] == 1:
                w += coeff * x_train[i]
    return w


def grad_desc(x_train, y_train, coeff, epochs):
    w = np.zeros(shape=len(x_train[0]))

    for epoch in range(epochs):
        for i in range(len(x_train)):
            f = sigmoid(x_train[i] @ w)
            w -= coeff * (f - y_train[i]) * f * (1 - f) * x_train[i]

    return w


def elementary_perceptron(x_train, y_train, numb_of_perceptrons, coeff, epochs, fit):
    x = np.hstack((np.ones((x_train.shape[0], 1)), x_train))

    x_trains = np.array_split(x, numb_of_perceptrons)
    y_trains = np.array_split(y_train, numb_of_perceptrons)

    weights = np.zeros(shape=(numb_of_perceptrons, len(x[0])))

    for i in range(numb_of_perceptrons):
        weights[i] = fit(x_trains[i], y_trains[i], coeff[i], epochs)

    return weights


def predict(w, activation_function, x):
    predicts = [activation_function(w[i][0] + (w[i][1:] @ x)) for i in range(len(w))]
    return Counter(predicts).most_common(1)[0][0]


def generate_confusion_matrix(y_actual, y_predicted):
    return metrics.confusion_matrix(y_actual, y_predicted).ravel()


iterations = 10000
number_of_samples = 500
train_rate = [0.03, 0.3, 1]
eps0 = 0.1

# For Gaussian
mu = np.array([1.0, 1.0])
sigma = np.array([1, 1])

# For Circle
radius = 10

# Ratio of training to test data
ratio = 50

start = -5
stop = 5
step = 0.1


def split_array(arr, q):
    # Рассчитываем индекс, по которому разделить массив
    split_index = int(len(arr) * q / 100)

    # Разделяем массив на две части
    arr_part1 = arr[:split_index]
    arr_part2 = arr[split_index:]

    return arr_part1, arr_part2


def show_data(x_train, y_train, color_1, color_2):
    x1 = [point[0] for point in x_train]
    x2 = [point[1] for point in x_train]
    colors = [color_1 if point == 0 else color_2 for point in y_train]
    plt.scatter(x1, x2, c=colors)


print("Enter number of perceptrons: ")
numb_of_perceptrons = int(input())
print("Enter 'g' for gaussian, 'c' for circle, 'x' for xor, 's' for spiral: ")
sample_type = input()
if sample_type == "g":
    features, targets = gaussian(number_of_samples, mu, sigma, eps0)
elif sample_type == "c":
    features, targets = circle(number_of_samples, radius, eps0)
elif sample_type == "x":
    features, targets = xor_data(number_of_samples, start, stop, eps0)
elif sample_type == "s":
    features, targets = spiral_data(number_of_samples, eps0)
else:
    raise NonExistingChoice

combined = list(zip(features, targets))
random.shuffle(combined)
trains, tests = split_array(combined, ratio)

x_train, y_train = zip(*trains)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test, y_test = zip(*tests)

x1 = np.concatenate((x_train[:, 0], x_train[:, 0]))
x2 = np.concatenate((x_train[:, 1], x_train[:, 1]))

start_time = time.time()
weights = elementary_perceptron(x_train, y_train, numb_of_perceptrons, train_rate, iterations, algorithm_studying)
end_time = time.time()

training_time = end_time - start_time
print('Training time: ', training_time)

answer = [predict(weights, step_function, x) for x in x_test]

tn, fp, fn, tp = generate_confusion_matrix(y_test, answer)
print(f"True positive: {tp}")
print(f"True negative: {tn}")

print(f"False positive: {fp}")
print(f"False negative: {fn}")

p = np.arange(start, stop, step)
f_p = [[f(weights[i], [p_i]) for p_i in p] for i in range(numb_of_perceptrons)]

plt.title('Training data')
show_data(x_train, y_train, "Orange", "Blue")
for i in range(numb_of_perceptrons):
    plt.plot(p, f_p[i], label=f'{i + 1}')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(min(x2) - 1, max(x2) + 1)
plt.grid(True)
plt.show()
