#################### SVM Q1) #########################

import pandas as pd
import numpy as np

data1 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Black box technique\SalaryData_Train (1).csv")

data2 = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Black box technique\SalaryData_Test (1).csv")
data = pd.concat([data1,data2], axis=0)

data = data.iloc[:,[13,0,1,2,3,4,5,6,7,8,9,10,11,12]]
data.drop(['race','native'], axis = 1, inplace = True )
data.info()
data.describe()
data.isnull().sum()
data[data.duplicated()]

# To convert the non numeric data to numeric data 
data.columns
data3 = pd.get_dummies(data, columns=['workclass', 'education','maritalstatus','occupation',  'relationship','sex'], drop_first = True)
data3.columns

# data normalisation 

def norm_fun(i):
    x = (i-i.min())/(i.max()- i.min())
    return(x)

data3_norm = norm_fun(data3.iloc[:,1:])


# split the data into train and test 

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(data3_norm,data.Salary ,test_size = 0.333)


from sklearn.svm import SVC

# kernel = linear 
model_linear = SVC( kernel = "linear")
model_linear.fit(train_X, train_Y)
pred_test_linear = model_linear.predict(test_X)

a = np.mean(pred_test_linear == test_Y)
a

pred_train_linear = model_linear.predict(train_X)
np.mean(pred_train_linear == train_Y)


# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_Y)
pred_test_rbf = model_rbf.predict(test_X)

b = np.mean(pred_test_rbf == test_Y)

# kernel = rbf
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X, train_Y)
pred_test_poly = model_poly.predict(test_X)

c = np.mean(pred_test_poly == test_Y)

# kernel = sigmoid

model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X,train_Y)
pred_test_sigmoid = model_sigmoid.predict(test_X)

d = np.mean(pred_test_sigmoid == test_Y)

comp = {"kernel":pd.Series(["linear","rbf","poly","sigmoid"]), "accuracy": pd.Series([a,b,c,d])}
comparision = pd.DataFrame(comp)
 y = comparision

# Linear kernel is giving the highest accuracy.

#############################################################################################

############################### Q2) ######################################################


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Black box technique\forestfires.csv")
data.info()
data.isnull().sum()
data[data.duplicated()]
data.describe()

data.drop(['month','day'], axis =1, inplace = True)

# data normalisation 

def norm_fun(i):
    x = (i-i.min())/(i.max()- i.min())
    return(x)

data_norm = norm_fun(data.iloc[:,0:28])


# split the data into train and test 

from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(data_norm,data.size_category ,test_size = 0.20, random_state = 50, shuffle = True)


from sklearn.svm import SVC

# kernel = linear 
model_linear = SVC( kernel = "linear")
model_linear.fit(train_X, train_Y)
pred_test_linear = model_linear.predict(test_X)

 a = np.mean(pred_test_linear == test_Y)
a
# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_Y)
pred_test_rbf = model_rbf.predict(test_X)

b = np.mean(pred_test_rbf == test_Y)
b
# kernel = rbf
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X, train_Y)
pred_test_poly = model_poly.predict(test_X)

c = np.mean(pred_test_poly == test_Y)
c
pred_test_poly = model_poly.predict(train_X)
np.mean(pred_test_poly == train_Y)

# kernel = sigmoid

model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X,train_Y)
pred_test_sigmoid = model_sigmoid.predict(test_X)

d = np.mean(pred_test_sigmoid == test_Y)
d


comp = {"kernel":pd.Series(["linear","rbf","poly","sigmoid"]), "accuracy": pd.Series([a,b,c,d])}
comparision = pd.DataFrame(comp)
 x = comparision

# poly kernel is giving the highest accuracy.