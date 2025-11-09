"""
Created on Wed Mar  5 16:44:39 2025
"""

import numpy as np
import pandas as pd
df = pd.read_csv(r"C:\Users\Jitu Patel\OneDrive\Desktop\New folder (4)\breast-cancer-wisconsin-data.csv""C:\Users\Jitu Patel\OneDrive\Desktop\New folder (4)\breast-cancer-wisconsin-data.csv")
df

df.info()
#------------------------------------------------------
# EDA
#------------------------------------------------------
# DATA CLEANING

#------------------------------------------------------
# dATA TRANSFORMTION
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder() 
df["diagnosis"] = LE.fit_transform(df["diagnosis"])
df["diagnosis"]
df.head()

df.shape

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X  = SS.fit_transform(df.iloc[:,2:])

X_new = pd.DataFrame(SS_X)
X_new.columns = list(df.iloc[:,2:])
X_new
#=====================================================
# PCA

from sklearn.decomposition import PCA
pca = PCA()

pcadata = pca.fit_transform(X_new)
pcadata = pd.DataFrame(pcadata)
pcadata


pcadata


#=====================================================
# Data partition
#=====================================================
X = pcadata.iloc[:,0:5]
Y = df["diagnosis"]

#==============================================================
# cross validation - shufflt split
#==============================================================
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


training_acc = []
test_acc = []

for i in range(1,101):
    X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.25,random_state=i)
    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(X_train,Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    training_acc.append(accuracy_score(Y_train,Y_pred_train))
    test_acc.append(accuracy_score(Y_test,Y_pred_test))
    
print("Cross validation: Training accuracy:",np.round(np.mean(training_acc),2))
print("Cross validation: Test accuracy:",np.round(np.mean(test_acc),2))
#==============================================================

















