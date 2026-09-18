from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder as OE
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

df=pd.read_csv("train.csv")
df["Deck"]=df["Cabin"].str[0]
df["family"]=df["SibSp"]+df["Parch"]
df.drop(columns=["PassengerId","Name","Ticket","SibSp","Parch","Cabin"],inplace=True)
print(df)

x_train,x_test,y_train,y_test=train_test_split(df.drop(columns="Survived"),df["Survived"],test_size=0.2)

trf1=ColumnTransformer([
    ("trf_emb_deck",SimpleImputer(strategy="most_frequent"),[4,5]),
    ("trf_age",SimpleImputer(),[2])
],remainder="passthrough")

trf2=ColumnTransformer([
    ("trf_sex", OE(categories=[["male", "female"]]), [4]),
    ("trf_deck_embark", OHE(drop="first"),[1,0])
],remainder="passthrough")

trf3=ColumnTransformer([
    ("trf_scaling",MinMaxScaler(),slice(0,14))
],remainder="passthrough")

trf4=SelectKBest(score_func=chi2,k=14)

trf5=DecisionTreeClassifier()

pipe =Pipeline([
    ('trf1',trf1),
    ('trf2',trf2),
    ('trf3',trf3),
    ('trf4',trf4),
    ('trf5',trf5)
])
# print(pipe)

pipe.fit(x_train,y_train)

y_pred= pipe.predict(x_test)

print(accuracy_score(y_pred,y_test))

import pickle

pickle.dump(pipe,open('pipe.pkl','wb'))