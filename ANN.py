import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

dataset = pd.read_csv('/Users/tharunpeddisetty/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv')

X = dataset.iloc[:,3:-1].values
Y = dataset.iloc[:,-1].values

#Label encoding for Gender column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2]=le.fit_transform(X[:, 2])  

#Encoding for country 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])] ,remainder='passthrough') #[0]: 0 is the index of the column
X=np.array(ct.fit_transform(X)) #forcing the output to be a numpy array. This is expected by the ML models


#Splitting data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,Y,test_size=0.2,random_state=0) #20% will go into testing set
print(X_train)

#Feature Scaling: Allows to put all our features into the same scale. Only certain ML models needs this.
#Feature scaling for NN is compulsory to be applied for all features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Building the model

#Initializing the ANN
#initializing ann as sequence of layers
# sequence class belongs to keras library but new tensorflow consists of keras as well
ann = tf.keras.models.Sequential()

#Adding input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu')) #units = # neurons in hidden layer. for fully function NN activation func=rectifier func or relu

#Add second hidden layer 
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#Add output layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid')) #for non binary output y more than 2 classifications use activation = 'softmax'

#Training ANN
# Compiling ANN
ann.compile(optimizer='adam' ,loss='binary_crossentropy' ,metrics=['accuracy']) #optimizer = stochastic gradient descent (adam) #for binary ouput classification we must use binary_crossentropy, non binary = categorical_crossentropy

#Training the ANN on training set
ann.fit(X_train,y_train,batch_size=32,epochs=100) #computing batches of data is more accurate. # of predictors

#Predicting the result of a single customer. Dont forget to scale the input and encode the input
ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))
#probability that cusomter leaves is above answer. 
#for true/false if customer leaves, we choose the threshold as 0.5 and print it 
print(ann.predict(sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]]))>0.5)


#Predicting test results. Code similar to logistic reg code
y_pred = ann.predict(X_test)
y_pred=(y_pred>0.5) #if >0.5 y_pred=1 else 0
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_pred),1)),1))

#Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)