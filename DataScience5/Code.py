import pandas as pd

train = pd.read_csv("train.csv")
#dropping collumns with na rows
train = train.dropna(axis= 1)

#parsing with label encoding for strings
numb = train.select_dtypes(exclude =['object'])
notnumb = train.select_dtypes(exclude =['int64','float64'])
notnumb = notnumb.drop('timestamp', axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
notnumble = notnumb.apply(le.fit_transform)

date = [None] * 30471
for i in range(30471):
    date[i] = [int(train['timestamp'][i][1:4]),int(train['timestamp'][i][5:7]), int(train['timestamp'][i][8:10])]
df = pd.DataFrame(date,columns=['year','month','day'])

train = numb.join(df)
train = train.join(notnumble)

#train splitting
from sklearn.model_selection import train_test_split

X = train.drop('price_doc', axis=1)
Y = train['price_doc']

X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.33, random_state=0)
#learning & results
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor()
clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))

from sklearn.metrics import mean_squared_log_error
print(mean_squared_log_error(Y_test,clf.predict(X_test)))