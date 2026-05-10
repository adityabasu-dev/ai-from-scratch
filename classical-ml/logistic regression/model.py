import numpy as np
import sklearn.datasets
import sklearn.model_selection
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def calculate_gradient(theta, X, y):
    m = y.size  #number of instances
    return (X.T @ (sigmoid(X@theta) - y) / m)

def gradient_descent(X, y, alpha=0.1, num_iter=1000, tol=1e-7):
    X_b =np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
    
    theta =np.zeros(X_b.shape[1])  # initialize theta

    for i in range(num_iter):
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha * grad  # update theta

        if np.linalg.norm(grad) < tol:
            print(f"Convergence reached at iteration {i}")
            break
    
    return theta
         
def predict_prob(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
    return sigmoid(X_b @ theta)  # return probability estimates

def predict(X, theta, threshold=0.5):
    return (predict_prob(X, theta) >= threshold).astype(int)  # return class labels

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# make some toy data
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
theta = gradient_descent(X_train, y_train, alpha=0.1, num_iter=1000)

# evaluate
preds = predict(X_test, theta)
accuracy = (preds == y_test).mean()
print(f"\nAccuracy: {accuracy:.3f}")