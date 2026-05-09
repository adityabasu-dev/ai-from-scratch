import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)

X = np.random.rand(50, 1) * 100  

Y = 3.5 * X + np.random.randn(50, 1) * 20

X = X.flatten()
Y = Y.flatten()

plt.scatter(X, Y)
plt.show()


def loss_function(m, b, X, Y):
    total_error = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(X))

def gradient_descent(m_now, b_now, X, Y, learning_rate):
    m_gradient = 0
    b_gradient = 0

    n=len(X)

    for i in range(n):
        x=X[i]
        y=Y[i]
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m_updated = m_now - learning_rate * m_gradient
    b_updated = b_now - learning_rate * b_gradient
    return m_updated, b_updated

m=0
b=0
learning_rate=0.0001
epochs=1000
for epoch in range(epochs):
    m, b = gradient_descent(m, b, X, Y, learning_rate)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: m={m}, b={b}, loss={loss_function(m, b, X, Y)}")
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, m * X + b, color='red', label='Regression Line')
plt.show()
print(f"Learned: m={m:.2f}, b={b:.2f}")
print(f"True:    m=3.5,  b=0")