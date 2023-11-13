# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5.Convert the text data into a numerical representation using CountVectorizer.
6.Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7.Finally, evaluate the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: S.Sakthi Priya
RegisterNumber: 212222040140 
*/
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
```

## Output:
Result Output
![s1](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104282/7115bb77-2a79-4728-b31b-cba6908a271a)


data.head()
![s2](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104282/f0407ba0-a899-43f2-9ec8-54033bb58bef)

data.info()

![s3](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104282/ed77ed17-2709-4721-8a46-f6285d4d646e)

data.isnull().sum()

![s4](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104282/8e687262-2415-4d46-9a17-de874d560c2c)

Y_Prediction Value
![s5](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104282/0ce13712-0372-4372-93f6-8c7b1f9a52e4)

Accuracy Value

![s6](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104282/9674fb2c-eab9-4842-82f8-b262c7826456)










## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
