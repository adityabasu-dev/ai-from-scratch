import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('placement.csv')

np.random.seed(42)

X = np.random.rand(50, 1) * 100  

Y = 3.5 * X + np.random.randn(50, 1) * 20
plt.scatter(X, Y)
plt.show()