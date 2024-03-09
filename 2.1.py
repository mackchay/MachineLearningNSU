import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

error_scale = 0.1

# Генерация случайных точек на плоскости для примера
num_samples = 300
x_1 = np.random.rand(num_samples, 2)

y_1 = (x_1[:, 0] + x_1[:, 1] > 1).astype(int)
x_2 = np.random.uniform(-1, 1, size=(num_samples, 2))
y_2 = (x_2[:, 0] ** 2 + x_2[:, 1] ** 2 < 0.5)
x_3 = x_2
y_3 = (x_3[:, 0] * x_3[:, 1] > 0)
t = np.random.uniform(0, 2 * np.pi, size=num_samples)
t_index = np.random.choice([-1, 1], size=num_samples)
r = 0.3
x_4 = np.vstack((t_index * r * t[:] * np.cos(t[:]), t_index * r * t[:] * np.sin(t[:]))).T
y_4 = (t_index[:] == 1)

# Создание объекта Dataset
dataset_1 = tf.data.Dataset.from_tensor_slices((x_1, y_1))
dataset_2 = tf.data.Dataset.from_tensor_slices((x_2, y_2))

# Вывод первого батча данных
for batch_x, batch_y in dataset_1.batch(32):
    print("Batch X shape:", batch_x.shape)
    print("Batch Y shape:", batch_y.shape)

# Визуализация данных на плоскости
fig, axes = plt.subplots(2, 2, figsize=(10, 10))


def create_axe(axe, x, y):
    axe.scatter(x[:, 0] + np.random.normal(0, error_scale, num_samples),
                x[:, 1] + np.random.normal(0, error_scale, num_samples), c=y, cmap='viridis')
    axe.set_xlabel("X_1-axis")
    axe.set_ylabel("X_2-axis")


create_axe(axes[0][0], x_1, y_1)
create_axe(axes[0][1], x_2, y_2)
create_axe(axes[1][0], x_3, y_3)
create_axe(axes[1][1], x_4, y_4)


def step_function(x):
    if x < 0:
        return 0
    else:
        return 1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def studying(x, y, horizon_coord, vert_coord):
    nu = 0.1
    w = np.zeros(3)
    for t in range(500):
        index = t % num_samples
        neuron_sum = w[0] * x[index][0] + w[1] * x[index][1] + w[2]

        neuron_output = sigmoid(neuron_sum)
        error = y[index] - neuron_output
        delta = error * sigmoid_derivative(neuron_output)

        w[0] += x[index][0] * nu * delta
        w[1] += x[index][1] * nu * delta
        w[2] += nu * delta

    x1_line = np.linspace(0, 1, 100)
    x2_line = - (w[0] / w[1]) * x1_line - (w[2] / w[1])

    axes[horizon_coord][vert_coord].plot(x1_line, x2_line, linewidth=1.5, color='red')


# studying(x_1, y_1, 0, 0)
# studying(x_2, y_2, 0, 1)
# studying(x_3, y_3, 1, 0)
# studying(x_4, y_4, 1, 1)

plt.show()
