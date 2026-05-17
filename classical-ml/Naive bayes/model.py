import numpy as np

class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing          # small value to prevent division by zero
        self.classes_ = None
        self.means_ = {}
        self.vars_ = {}
        self.priors_ = {}
    
    # Fit the model to the training data
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # Calculate mean, variance, and prior for each class
        for cls in self.classes_:
            X_cls = X[y == cls]
            self.means_[cls] = X_cls.mean(axis=0)
            self.vars_[cls] = X_cls.var(axis=0) + self.var_smoothing
            self.priors_[cls] = X_cls.shape[0] / X.shape[0]
    
    # Calculate the Gaussian probability density function
    def gaussian_pdf(self, x, mean, var):
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(-0.5 * ((x - mean) ** 2) / var)
        return coeff * exponent
    
    #predict class labels for the input data
    def predict(self, X):
        preds = []
        for x in X:
            class_probs = {}
            for cls in self.classes_:
                mean = self.means_[cls]
                var = self.vars_[cls]
                prior = self.priors_[cls]
                log_likelihood = np.sum(np.log(self.gaussian_pdf(x, mean, var)))   # calculate log likelihood to prevent underflow
                class_probs[cls] = np.log(prior) + log_likelihood
            preds.append(max(class_probs, key=class_probs.get))
        return np.array(preds)
    


#Example usage
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(f"Accuracy: {(preds == y_test).mean():.3f}")