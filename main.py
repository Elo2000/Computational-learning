import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# # קטע קוד לטריינינג והטסט של המודלים
start_time = time.time()

# Load the dataset from CSV
data = pd.read_csv('creditcard.csv')

# Split the data into features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Decision Tree class for building the model
class DecisionTree:
    def _init_(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        node = {'predicted_class': predicted_class}

        if depth < self.max_depth:
            feature, threshold = self._best_split(X, y)
            if feature is not None:
                indices_left = X[:, feature] < threshold
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature'] = feature
                node['threshold'] = threshold
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree
        while 'feature' in node:
            if inputs[node['feature']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']

# Adaboost class for ensemble learning
class Adaboost:
    def _init_(self, num_estimators=50):
        self.num_estimators = num_estimators
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        sample_weights = np.full(n_samples, (1 / n_samples))

        for _ in range(self.num_estimators):
            estimator = DecisionTree(max_depth=1)
            estimator.fit(X, y)
            predictions = estimator.predict(X)

            error = np.sum(sample_weights[predictions != y])
            estimator_weight = 0.5 * np.log((1 - error) / max(error, 1e-16))

            sample_weights *= np.exp(-estimator_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)

            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for estimator, estimator_weight in zip(self.estimators, self.estimator_weights):
            predictions += estimator_weight * np.array(estimator.predict(X))

        return np.sign(predictions)

# Create and train the Logistic Regression classifier for each class using One-vs-All
classifiers = {}
for class_label in np.unique(y_train):
    y_train_binary = (y_train == class_label).astype(int)
    y_test_binary = (y_test == class_label).astype(int)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train_binary)

    classifiers[class_label] = logreg


# # Predictions using One-vs-All
def predict_one_vs_all(classifiers, X):
    predictions = []
    for _, clf in classifiers.items():
        pred = clf.predict(X)
        predictions.append(pred)
    return np.array(predictions).T.argmax(axis=1)

# Predict using One-vs-All approach for Logistic Regression
logreg_predictions = predict_one_vs_all(classifiers, X_test)

# Print metrics for One-vs-All Logistic Regression
print("One-vs-All Logistic Regression Metrics:")
print("Accuracy:", accuracy_score(y_test, logreg_predictions))
print("Precision:", precision_score(y_test, logreg_predictions, average='weighted'))
print("Recall:", recall_score(y_test, logreg_predictions, average='weighted'))
print("F1 Score:", f1_score(y_test, logreg_predictions, average='weighted'))

# Implement and train Adaboost model
adaboost = Adaboost()
adaboost.fit(X_train.values, y_train.values)
adaboost_predictions = adaboost.predict(X_test.values)

# Print metrics for Adaboost
print("Adaboost Metrics:")
print("Accuracy:", accuracy_score(y_test, adaboost_predictions))
print("Precision:", precision_score(y_test, adaboost_predictions))
print("Recall:", recall_score(y_test, adaboost_predictions))
print("F1 Score:", f1_score(y_test, adaboost_predictions))

# Evaluate Logistic Regression on the training set
logreg_train_predictions = logreg.predict(X_train)
print("Logistic Regression Metrics on Training Set:")
print("Accuracy:", accuracy_score(y_train, logreg_train_predictions))
print("Precision:", precision_score(y_train, logreg_train_predictions))
print("Recall:", recall_score(y_train, logreg_train_predictions))
print("F1 Score:", f1_score(y_train, logreg_train_predictions))

# Evaluate Adaboost on the training set
adaboost_train_predictions = adaboost.predict(X_train.values)
print("Adaboost Metrics on Training Set:")
print("Accuracy:", accuracy_score(y_train, adaboost_train_predictions))
print("Precision:", precision_score(y_train, adaboost_train_predictions))
print("Recall:", recall_score(y_train, adaboost_train_predictions))
print("F1 Score:", f1_score(y_train, adaboost_train_predictions))

# Cross-validation for Logistic Regression
logreg_cv_scores = cross_val_score(logreg, X_train, y_train, cv=5)
print("Logistic Regression Cross-Validation Scores:", logreg_cv_scores)
print("Average Cross-Validation Accuracy:", np.mean(logreg_cv_scores))

# Cross-validation for Adaboost
adaboost_cv_scores = cross_val_score(adaboost, X_train.values, y_train.values, cv=5)
print("Adaboost Cross-Validation Scores:", adaboost_cv_scores)
print("Average Cross-Validation Accuracy:", np.mean(adaboost_cv_scores))

# Plot the comparison of Logistic Regression and Adaboost
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
logreg_metrics = [accuracy_score(y_test, logreg_predictions),
                  precision_score(y_test, logreg_predictions),
                  recall_score(y_test, logreg_predictions),
                  f1_score(y_test, logreg_predictions)]
adaboost_metrics = [accuracy_score(y_test, adaboost_predictions),
                    precision_score(y_test, adaboost_predictions),
                    recall_score(y_test, adaboost_predictions),
                    f1_score(y_test, adaboost_predictions)]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, logreg_metrics, width, label='Logistic Regression')
rects2 = ax.bar(x + width/2, adaboost_metrics, width, label='Adaboost')

ax.set_ylabel('Scores')
ax.set_title('Comparison of Logistic Regression and Adaboost')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()

plt.show()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds")S