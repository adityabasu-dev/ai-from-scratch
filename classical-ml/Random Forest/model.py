import numpy as np
from collections import Counter

def gini(y):
    m= len(y)
    if m == 0:
        return 0
    count = Counter(y)
    impurity = 1.0
    for lbl in count:
        prob_of_lbl = count[lbl] / m
        impurity -= prob_of_lbl ** 2
    return impurity

class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature   = feature
        self.threshold = threshold
        self.left      = left
        self.right     = right
        self.value     = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features = n_features  # number of features to consider for splits

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples = len(y)

        # stopping conditions
        # 1. max depth reached
        # 2. not enough samples to split
        # 3. all samples have the same label
        if (self.max_depth is not None and depth >= self.max_depth) \
                or n_samples < self.min_samples_split \
                or len(set(y)) == 1:
            return Node(value=self._leaf_value(y))

        feat, thresh = self._best_split(X, y)  # find the best split

        if feat is None:                        # no useful split found
            return Node(value=self._leaf_value(y))

        left_mask  = X[:, feat] <= thresh
        right_mask = ~left_mask

        left  = self._build_tree(X[left_mask],  y[left_mask],  depth+1)
        right = self._build_tree(X[right_mask], y[right_mask], depth+1)
        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def _leaf_value(self, y):
        return Counter(y).most_common(1)[0][0]  # majority class
    
    def _best_split(self, X, y):
        best_gain = 0
        best_feat, best_thresh = None, None
        n_features = self.n_features or X.shape[1]  # default: use all
        feats = np.random.choice(X.shape[1], n_features, replace=False)

        
        for feat in feats:
            for thresh in np.unique(X[:, feat]):
                # split y into left and right
                # compute IG
                # update best if IG > best_gain
                pred_Left = y[X[:, feat] <= thresh]
                pred_Right = y[X[:, feat] > thresh]
                gain = gini(y) - (len(pred_Left) / len(y)) * gini(pred_Left) - (len(pred_Right) / len(y)) * gini(pred_Right)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh
    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])
    
    def _predict(self, inputs, node):
        if node.is_leaf():
            return node.value
        if inputs[node.feature] <= node.threshold:
            return self._predict(inputs, node.left)
        else:
            return self._predict(inputs, node.right)
        
    




class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, 
                 min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # number of features to consider for each split
        self.trees = []               # list to store trained trees

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # bootstrap sampling
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            # train a decision tree on the bootstrap sample
            n_features = self.n_features or int(np.sqrt(X.shape[1])) # default: sqrt of total features
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_features=n_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Get predictions from each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return the majority vote
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] 
                     for i in range(X.shape[0])])
    


# For testing the implementation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# load data
data = load_iris()
X, y = data.data, data.target

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train
rf = RandomForest(n_trees=10, max_depth=5)
rf.fit(X_train, y_train)

# predict
preds = rf.predict(X_test)

# accuracy
accuracy = np.mean(preds == y_test)
print(f"Accuracy: {accuracy:.2%}")