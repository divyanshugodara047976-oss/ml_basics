from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder as OE
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import LabelEncoder as LE
import pandas as pd
import numpy as np

df=pd.read_csv("customer.csv")
# print(df.sample())

X_train,X_test,Y_train,Y_test=train_test_split(df[["review"]],df[["purchased"]],test_size=0.2)
x_train,x_test=train_test_split(df[["age"]],test_size=0.2)
oe=OE(categories=[["Poor","Average","Good"]])
oe.fit(X_train)
X_train=oe.transform(X_train)
X_test=oe.transform(X_test)
# print(X_train)
Z_train,Z_test=train_test_split(df[["gender","education"]],test_size=0.2)
ohe=OHE()
ohe.fit(Z_train)
Z_train=ohe.transform(Z_train).toarray()
Z_test=ohe.transform(Z_test).toarray()
X_train=np.hstack((X_train,x_train,Z_train))
X_test=np.hstack((x_test,X_test,Z_test))
# print(X_train)
le=LE()
le.fit(Y_train)

y_train=le.transform(Y_train)
y_test=le.transform(Y_test)
# print(Y_train)
model = LogisticRegression(class_weight="balanced")
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Detailed report
print("Classification Report:\n", classification_report(y_test, y_pred))






