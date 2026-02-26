import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset=pd.read_csv("diabetes.csv")

dataset = dataset.drop_duplicates()

X=dataset.drop(columns=["Outcome"])
Y=dataset["Outcome"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(X_train,Y_train)

predictions=model.predict(X_test)
accuracy=accuracy_score(Y_test,predictions)
print(accuracy)

prediction1=model.predict([[1, 95, 70, 20, 85, 24.5, 0.2, 22]])
if prediction1:
    print("Diabetes Present")
else:
    print("Not Present")
