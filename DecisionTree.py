# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:07:32 2024

@author: DELL
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt   

#veri seti inceleme
iris = load_iris()

X = iris.data# features
y = iris.target# target

X_train,X_test,y_train,y_test = train_test_split(X, y,test_size=0.2,random_state = 42)

#Dt modeli oluştur ve train et
tree_clf = DecisionTreeClassifier(criterion='entropy',max_depth=5, random_state=42) #criterion = 'entropy'
tree_clf.fit(X_train, y_train)

#DT evaluation test
y_pred= tree_clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("iris veri seti ile egitilen DT modeli doğruluğu: " ,accuracy)

conf_matrix = confusion_matrix(y_test,y_pred)
print("conf_matrix")
print(conf_matrix)

plt.figure(figsize=(15,10))
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names,class_names=list(iris.target_names))
plt.show()


feature_importance = tree_clf.feature_importances_