import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20
columnNames = ["ID","Malicious"]
columnNames.extend(range(1,532))
virus = pd.read_csv('train.csv', names = columnNames)
X, y = virus.drop('Malicious',axis=1).drop('ID',axis=1)[1:].as_matrix(), virus['Malicious'][1:].values
dep = range(1,10)

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step
for z in dep:
    classifiers = [("DecissionTree: gini random "+ str(z), tree.DecisionTreeClassifier(criterion='gini',splitter='random',max_depth=z))]

    xx = 1. - np.array(heldout)

    for name, clf in classifiers:
        print("training %s" % name)
        rng = np.random.RandomState(42)
        yy = []
        for i in heldout:
            yy_ = []
            for r in range(rounds):
                start = timeit.default_timer()
                X_train, X_test, y_train, y_test = \
                    train_test_split(X, y, test_size=i, random_state=rng)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                stop = timeit.default_timer()
                #prediction_dataframe = pd.DataFrame(data=y_pred, index=y_test.index, columns=['Malicious'])
                #test_dataframe = pd.DataFrame(data=y_test, index=y_test.index, columns=['Malicious'])
                #prediction_dataframe.to_csv("solution-" + str(i) + "-" + str(r) + ".csv")
                yy_.append(1 - np.mean(y_pred == y_test))
            yy.append(np.mean(yy_))
        plt.plot(xx, yy, label=name + " " + str(stop - start) + "s")

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()
