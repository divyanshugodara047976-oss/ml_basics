train,x_test,y_train,y_test=train_test_split(df.drop(columns="Survived"),df["Survived"],test_size=0.2)

# trf1=ColumnTransformer([
#     ("trf_emb_deck",SimpleImputer(strategy="most_frequent"),[4,5]),
#     ("trf_age",SimpleImputer(),[2])
# ],remainder="passthrough")

# trf2=ColumnTransformer([
#     ("trf_sex_deck_embark", OE(categories=[["male", "female"],["S","Q","C"],["A","B","C"]]), [4])
# ],remainder="passthrough")

# trf3=ColumnTransformer([
#     ("trf_scaling",MinMaxScaler(),slice(0,6))
# ],remainder="passthrough")

# trf4=SelectKBest(score_func=chi2,k=6)

# trf5=DecisionTreeClassifier()

# pipe =Pipeline([
#     ('trf1',trf1),
#     ('trf2',trf2),
#     ('trf3',trf3),
#     ('trf4',trf4),
#     ('trf5',trf5)
# ])
# # print(pipe)

# pipe.fit(x_train,y_train)

# y_pred= pipe.predict(x_test)

# print(accuracy_score(y_pred,y_test))

