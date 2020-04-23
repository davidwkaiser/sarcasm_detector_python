from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#from scipy.io import arff #do we need?
import pandas as pd

#Read the data from file into a data frame
df = pd.read_csv("~/Desktop/DePaul_DataScience_Credential/Data_Fundamentals/train-balanced-sarcasm.csv")

#print(df.head())

# Sample it, the data set is huge
sample_size = 0.05
seed = 123
df1_5 = df.sample(frac = sample_size, random_state = seed)


# Subset it so we only use the variables we need
xLabel = ["comment"]
yLabel = ["label"]
	
#print(df1_5.head())

df2 = df1_5[xLabel]
yVar = df1_5.iloc[:,0]

#print(df2.head())
#print(yVar.head())
#print(len(df2))


X_train, X_test, y_train, y_test = train_test_split(df2, yVar, test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


cvec = CountVectorizer()

X_train = cvec.transform(X_train)

X_test = cvec.transform(X_test)

rf = RandomForestClassifier(n_jobs=2, random_state=0)

rf.fit(X_train, y_train)

rf.score(X_train, y_train)

rf.score(test, y_test)





#preds = clf.predict(X_test)

#print(pd.crosstab(y_test, preds, rownames=['Actual Result'], colnames=['Predicted Result']))
