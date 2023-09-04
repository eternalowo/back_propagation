import numpy as np

w1 = np.array([[0.2, 0.5],
               [0.3, 0.6],
               [0.1, -0.5]])

w2 = np.array([[0.2, -0.3, 0.5],
               [0.6, -0.2, -0.3]])

w3 = np.array([-0.2, 0.4])


def f(x):
    return 2 / (1 + np.exp(-x)) - 1


def df(x):
    return 0.5 * (1 + x) * (1 - x)


def activate_neural(x):

    v1 = np.dot(w1, x)
    f1 = np.array([f(i) for i in v1])

    v2 = np.dot(w2, f1)
    f2 = np.array([f(i) for i in v2])

    v_out = np.dot(w3, f2)
    y = f(v_out)
    return f1, f2, y


def back_propagation(train_data, correct, n=10000, lmb=0.01):
    global w1, w2, w3
    cnt = len(train_data)

    for _ in range(n):
        ind = np.random.randint(0, cnt)
        x = train_data[ind]
        out = activate_neural(x)

        err = out[2] - correct[ind]
        delta3 = err * df(out[2])
        w3 = w3 - lmb * delta3 * out[1]

        delta2 = w3 * delta3 * df(out[1])
        w2[0, :] = w2[0, :] - lmb * delta2[0] * out[0]
        w2[1, :] = w2[1, :] - lmb * delta2[1] * out[0]

        b1 = np.dot(w2[:, 0], delta2)
        b2 = np.dot(w2[:, 1], delta2)
        b3 = np.dot(w2[:, 2], delta2)
        delta1 = np.array([b1, b2, b3]) * df(out[0])
        w1[0, :] = w1[0, :] - lmb * delta1[0] * np.array(x)
        w1[1, :] = w1[1, :] - lmb * delta1[1] * np.array(x)
        w1[2, :] = w1[2, :] - lmb * delta1[2] * np.array(x)


x = [[-1, 1],
     [1, 1],
     [-1, -1],
     [1, -1]]
y = [1, -1, 1, 1]

back_propagation(x, y)

for i in x:
    print(activate_neural(i)[-1])
