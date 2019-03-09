from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

noticias = fetch_20newsgroups(subset="train")
noticias.data[0]
len(noticias.data)
noticias.target_names
vector=CountVectorizer()
vector.fit(noticias.data)
#print(vector.vocabulary_)
bolsa=vector.transform(noticias.data)
bolsa.shape
bolsay=noticias.target
Xe, Xt, ye, yt=train_test_split(bolsa,bolsay)
lr=LogisticRegression()
lr.fit(Xe,ye)
print(lr.score(Xt, yt))
