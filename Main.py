import pandas as  pd 
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sb 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import time




data = pd.read_csv('data.txt')
print("\n \t The data frame has {0[0]} rows and {0[1]} columns.\n" .format(data.shape))
#data.info()
#print(data.head(3))



# Counting malign and benign in data
diagnosis_all = list(data.shape)[0]
diagnosis_categories = list(data['M'].value_counts())
print("\n \t The data has {} diagnosis {} malignant and {} benign data \n" . format(diagnosis_all, diagnosis_categories[0], diagnosis_categories[1]))

#Data visualization
features_mean= list(data.columns[1:11])
plt.figure(figsize = (10, 10))
#sns.heatmap(data[features_mean])
sb.heatmap(data[features_mean].corr(), annot = True, square = True, cmap = 'coolwarm')

plt.show()
# scatterplot


color_dic = {'M':'red', 'B':'blue'}
colors = data['M'].map(lambda x: color_dic.get(x))

sm = pd.scatter_matrix(data[features_mean], c=colors, alpha=0.4, figsize=((15,15)));

plt.show()

# plotting different graph for different features

bins = 12
plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    
    plt.subplot(rows, 2, i+1)
    
   #sb.distplot(data[data['M']=='M'][feature], bins=bins, color='red', label='M');
    #sb.distplot(data[data['B']=='M'][feature], bins=bins, color='blue', label='B');
    
    plt.legend(loc='upper right')

plt.tight_layout()
plt.show()



# boxplot

plt.figure(figsize=(15,15))
for i, feature in enumerate(features_mean):
    rows = int(len(features_mean)/2)
    
    plt.subplot(rows, 2, i+1)
    
   # sb.boxplot(x='M', y=feature, data=data, palette="Set1")

plt.tight_layout()
plt.show()


features_selection = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']

diag_map = {'M':1, 'B':0}
data['M'] = data['M'].map(diag_map)
 # seperating train and test data, train is 20% of data
X = data.loc[:, features_mean]
\
y = data.loc[:, 'M']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

accuracy_all = []
cvs_all = []

# Using Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
start = time.time();
clf = SGDClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf,X,y,cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
print("SGD Classifier Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


# SVM 
# SVC

from sklearn.svm import SVC, NuSVC, LinearSVC

start = time.time();
clf = SVC();
clf.fit(X_train, y_train)

prediction = clf.predict(X_test)
scores = cross_val_score(clf,X,y,cv=5)
end = time.time()
accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
print("SVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

#NuSVC
start = time.time();
clf = NuSVC()
clf.fit(X_train,y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv= 5)
end = time.time();
accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
print("NuSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


#LinearSVC
start = time.time();
clf = LinearSVC();
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)
end  = time.time()
accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))
print("LinearSVC Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))


#Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
start = time.time()
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv= 5)
end = time.time()
accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Nearest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))

# Naive Bayes 

from sklearn.naive_bayes import GaussianNB
start = time.time()
clf = GaussianNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv= 5)
end = time.time()
accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Naive Bayes Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: {0:.5} seconds \n".format(end-start))



#random frorest

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

start = time.time()

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Random Forest Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start = time.time()

clf = ExtraTreesClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Extra Trees Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))

start = time.time()

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)

end = time.time()

accuracy_all.append(accuracy_score(prediction, y_test))
cvs_all.append(np.mean(scores))

print("Dedicion Tree Accuracy: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Cross validation score: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Execution time: %s seconds \n" % "{0:.5}".format(end-start))





















