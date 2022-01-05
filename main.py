from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

boston = datasets.load_boston()

X = boston.data
y = boston.target

print(X)
print(X.shape)
print(y.shape)
#algorihm
l_reg = linear_model.LinearRegression()

plt.scatter(X.T[5], y)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#train
model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)
print("R^2: ", l_reg.score(X,y))
print("coedd: ", l_reg.coef_)