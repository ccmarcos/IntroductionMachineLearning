from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import metrics

iris = datasets.load_iris()
X = iris.data
y = iris.target

km = KMeans(n_clusters=3,max_iter=3000)
km.fit(X)
prediccions=km.predict(X)
score=metrics.adjusted_rand_score(y,prediccions)
print(score)