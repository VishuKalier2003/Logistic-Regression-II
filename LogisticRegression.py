import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("C:/Users/VISHU/3D Objects/Machine Learning/Supervised Learning/Regression/train_and_test2.csv")
print("Shape with Null values : ",data.shape)
data = data.dropna()
data.replace(to_replace = np.nan, value=0)
print("Shape with null values removed : ",data.shape)
print("Data sample : ",data.head(4))
x = data[['Fare']]
y = data[['Sex']]

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)
print("Training data size : ",xtrain.shape)
print("Test data size : ",xtest.shape)

from sklearn.preprocessing import StandardScaler
standardscalar = StandardScaler()
xtrain = standardscalar.fit_transform(xtrain)
xtest = standardscalar.transform(xtest)

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(random_state=0)
logistic.fit(xtrain, ytrain)

ypred = logistic.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, ypred)
print("The Confusion matrix for testing and predicting dataset : ",cm)

from sklearn.metrics import accuracy_score
print(accuracy_score(ypred, ytest)*100,'%')

xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
xtest = np.array(xtest)
ytest = np.array(ytest)
ypred = np.array(ypred)
xt = xtest.reshape((327, 1))
yt = ytest.reshape((327, 1))
yp = ypred.reshape((327, 1))
xtr = xtrain.reshape((980, 1))
ytr = ytrain.reshape((980, 1))

plt.figure(figsize=(12, 8))
plt.scatter(xt, yt, color='blue')
plt.scatter(xt, yp, color='red')
plt.xlabel("X features (independent)", color='red', size=16)
plt.ylabel("Y features (dependent)", color='red', size=16)
plt.title("Distribution of Data using Logistic Regression", color='blue', size=20)
sns.regplot(xt, yt, data=data, ci=None, scatter_kws={'color': 'none'})
plt.legend(['Actual Test Values', 'Predicted Test Values'])
plt.show()