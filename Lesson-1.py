# sklearn ML Library
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
# Veri seti incelemesi

cancer = load_breast_cancer()

df = pd.DataFrame(data= cancer.data, columns= cancer.feature_names)
df["target"] = cancer.target  

# Makine Öğrenmesi Modelinin secilmesi KNN Sınıflandırıcı
# Modelin train edilmesi

X= cancer.data# feature
y= cancer.target#target

#trrain test split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size = 0.3, random_state = 42)

#olceklendirme

scaller= StandardScaler()
X_train = scaller.fit_transform(X_train)
X_test = scaller.transform(X_test)





# knn model oluştu ve tran edildi

knn = KNeighborsClassifier(n_neighbors=3)    # model oluşturma komsu parametresini unutma

knn.fit(X_train,y_train) #fit fonsiyonu verimizi (saple+target) kullanarak knn algoritmasını eğitir

#Sonuçların değerlendirilmesi :test
y_predict = knn.predict(X_test)

accuracy = accuracy_score(y_test,y_predict)

print("Doğruluk:",accuracy)

conf_matrix = confusion_matrix(y_test,y_predict)
print("confusion_matrix:")
print(conf_matrix)
#Hiper Parametre ayarlaması

"""
KNN :Hyyper parameter = K
    K:1,2,3 ...N
    Accuracy: %A, %B ...
        

"""
k=10
knn= KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)
y_prep = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_prep)
print(accuracy)




























