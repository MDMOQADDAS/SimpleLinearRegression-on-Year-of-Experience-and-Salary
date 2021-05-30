import pandas
dataset = pandas.read_csv("SalaryData.csv")
X =dataset['YearsExperience']
y = dataset['Salary']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train = X_train.values.reshape(24 , 1)
y_train = y_train.values.reshape(24 , 1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train , y_train)
model.predict([[10]])
