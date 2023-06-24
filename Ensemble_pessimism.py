# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
# %matplotlib inline

df = pd.read_excel('data.xlsx')
df.rename(columns=df.iloc[0], inplace=True)
df = df.drop(df.index[0])
df = df.dropna(subset=['label'])
df = df.dropna(axis=1)
df = df.drop(columns=['번호','날짜','시간','시편상태','요약','수막두께'])
df = df.apply(pd.to_numeric)
df.columns

X = df.drop('label',axis=1).values
y = df['label'].values

from sklearn.model_selection import train_test_split


# +
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# +
######## LogisticRegression ########

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions_1 = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions_1))
print(classification_report(y_test, predictions_1))

# +
##### Support Vector Machine #####

from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)

predictions_2 = svc_model.predict(X_test)
print(confusion_matrix(y_test, predictions_2))
print(classification_report(y_test, predictions_2))

# +
##### SGD Classifier #####

from sklearn.linear_model import SGDClassifier

sc=SGDClassifier(loss='log',max_iter=100,random_state=42)
sc.fit(X_train,y_train)
classes=np.unique(y_train)

train_score=[]
test_score=[]

for _ in range(0,300):
    sc.partial_fit(X_train,y_train,classes=classes)
    train_score.append(sc.score(X_train,y_train))
    test_score.append(sc.score(X_test,y_test))

predictions_3 = sc.predict(X_test)

print(confusion_matrix(y_test, predictions_3))
print(classification_report(y_test, predictions_3))


import matplotlib.pyplot as plt

fig,ax1=plt.subplots()
color_1='tab:blue'
ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.plot(train_score,color=color_1)

ax2=ax1.twinx()
color_2='tab:red'
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.plot(test_score,color=color_2)

plt.show()


# +
##### Ensemble Hard Voting #####

predictions=[]

for i in range(len(predictions_1)):
    if (((predictions_1[i]) + (predictions_2[i]) + (predictions_3[i]))>=1):
        predictions.append(1)
    else: predictions.append(0)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# +
import tensorflow as tf
from tensorflow import keras

model=keras.Sequential()
model.add(keras.layers.Dense(128,activation='relu',input_shape=(35,)))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
model.summary()

history=model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test))

result=model.predict_classes(X_test)



# -

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, result))
print(classification_report(y_test, result))

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

history_dict = history.history
print(history_dict.keys())

# +
from sklearn import metrics

def plot_roc_curve(fper, tper):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.show()
prob=sc.predict_proba(X_test)
prob=prob[:,1]
fper,tper,thresholds=metrics.roc_curve(y_test,prob)
plot_roc_curve(fper,tper)
# -

prob = model.predict(X_test)
prob = prob[:]
fper, tper, thresholds = metrics.roc_curve(y_test, prob)
plot_roc_curve(fper, tper)


