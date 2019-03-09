from sklearn.externals import joblib
from sklearn import datasets

clf = joblib.load('modelo_entrenado.pkl')
iris=datasets.load_iris()
print(clf.score(iris.data, iris.target))