import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X = pd.DataFrame(iris.data)
y = pd.DataFrame(iris.target)

# set aside 50% of train and test data for evaluation
# don't change random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=123)

print(X_train.shape, y_train.shape)


def predict(n):
    knn = KNeighborsClassifier(n_neighbors=n, weights='distance')
    knn.fit(X_train, y_train.values.ravel())
    y_pred = knn.predict(X_test)

    print(f'N: {n} Accuracy: {metrics.accuracy_score(y_test, y_pred)}')


for n in [5, 6, 7, 8, 9]:
    predict(n)
