import pandas as pd
import numpy as np

#reading & parsing
train = pd.read_csv("train.csv")

from sklearn.model_selection import train_test_split

date = [None] * 10886
for i in range(10886):
    date[i] = [int(train['datetime'][i][5:7]), int(train['datetime'][i][8:10]), int(train['datetime'][i][11:13])]
df = pd.DataFrame(date,columns=['month','day','time'])
train = train.join(df)

#test train spitting
X = train.drop(['datetime','casual', 'registered', 'count'], axis = 1)
y = train['count']

X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=0)

#tree is growing
from sklearn import tree
clf = tree.DecisionTreeRegressor(max_depth=11, min_samples_split=3)
clf.fit(X_train,y_train)


#printing scores and grid scores
print(clf.score(X_test, y_test))

from sklearn.metrics import f1_score
y_pred = np.array(clf.predict(X_test), dtype = int)
print(f1_score(y_test , y_pred, average='weighted'))

tree.export_graphviz(clf, out_file='tree.dot')
