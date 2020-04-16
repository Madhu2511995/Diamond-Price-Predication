import pandas as pd
import numpy as np

# Read the data
diamond=pd.read_csv("diamonds.csv",index_col=0)
diamond.head(5)
# Get the information about the dataset
diamond.info()

#Convert the object datatype into the numerical datatype
cut_class_dict={'fair':1,'Good':2, 'Very Good':3,'Premium':4,'Ideal':5}
clarity_dict={'I1':1,'SI2':2, 'SI1':3,'VS2':4, 'VS1':5, 'VVS2':6, 'VVS1':7,'IF':8}
color_dict={'J':1,'I':2,'H':3,'G':4,'F':5,'E':6,'D':7}

diamond['cut']=diamond['cut'].map(cut_class_dict)
diamond['color']=diamond['color'].map(color_dict)
diamond['clarity']=diamond['clarity'].map(clarity_dict)

#Check the missing value in the dataset
diamond.isnull().sum()
# fill the missing value
diamond['cut'].fillna('0',inplace=True)


#Create the test and train variable
import sklearn
from sklearn import preprocessing
diamond=sklearn.utils.shuffle(diamond)
x=diamond.drop('price',axis=1).values
x=preprocessing.scale(x)
y=diamond['price'].values

test_size=200
x_train=x[:-test_size]
y_train=y[:-test_size]
x_test=x[-test_size:]
y_test=y[-test_size:]

# Create first Machine-Learning-Model
from sklearn import svm
clf=svm.SVR(kernel='linear')
clf.fit(x_train,y_train)
print('Accuracy :',clf.score(x_test,y_test))

# Create Second Machine-Learning-Model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print("Accuracy Of Model Using Train Dataset:",lr.score(x_train,y_train))
print("Accuracy Of Model Using Test Dataset:",lr.score(x_test,y_test))


#Create a zip file for all test dataset
for x,y in zip(x_test,y_test): 
    print(f"Model :{lr.predict([x])[0]},Actual :{y}")
