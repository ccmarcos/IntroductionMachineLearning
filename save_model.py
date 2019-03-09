from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
clf=linear_model.LogisticRegression()
iris.keys()
XE, XT, ye, yt=train_test_split(iris.data, iris.target)
clf.fit(XE,ye)
print(clf.score(XT, yt))

from sklearn.externals import joblib
joblib.dump(clf, 'modelo_entrenado.pkl')