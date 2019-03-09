import numpy as np 
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
type(iris)
print(iris.keys())
#print(iris['data'])
print(iris['target_names'])
print(iris['target'])
#print(iris['feature_names'])
X_train, X_test, y_train, y_test = train_test_split(iris['data'],iris['target'])
print(y_train.shape)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)
print(knn.score(X_test,y_test))
print(knn.predict([[1.2,3.4,5.6,1.1]]))