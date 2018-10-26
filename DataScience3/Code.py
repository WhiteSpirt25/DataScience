import pandas as pd
from sklearn.metrics import mean_squared_error

def splitAndFit(X, y, regr):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)

    regr.fit(X_train, y_train)

    # printing scores
    print(clf.score(X_test, y_test))
    print(f'MSE: {mean_squared_error(y_test, clf.predict(X_test)):.2f}')


#parsing
train = pd.read_csv("train.csv")

from sklearn import linear_model
clf = linear_model.LinearRegression()

from sklearn.model_selection import train_test_split
#превращаем строку с датой в столбцы месяц, день и время(часы)
date = [None] * 10886
for i in range(10886):
    date[i] = [int(train['datetime'][i][5:7]), int(train['datetime'][i][8:10]), int(train['datetime'][i][11:13])]
df = pd.DataFrame(date,columns=['month','day','time'])
train = train.join(df)

#test train spitting & fitting
X = train.drop(['datetime','casual', 'registered', 'count'], axis = 1)
y = train['count']

print('Without one hot or label encoding')
splitAndFit(X,y,clf)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Xle = X.apply(le.fit_transform)
print('With Label encoding')
splitAndFit(Xle,y,clf)

#one hot encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore', sparse='false')
enc.fit(X)#нет слысла в Xle так как разницы в результате нет
onehot = enc.transform(X)

print('One hot encoded')
splitAndFit(onehot,y,clf)
