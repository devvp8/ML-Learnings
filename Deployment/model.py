import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
df=pd.read_csv("C:\\Users\\Dev Atul Patel\\OneDrive\\Documents\\language_dataset.csv")
x=np.array(df['text'])
y=np.array(df['language'])

cv=CountVectorizer()
X=cv.fit_transform(x)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=41)

model=MultinomialNB()
model.fit(X_train,y_train)
model.score(X_test,y_test)

pred=model.predict(X_test)