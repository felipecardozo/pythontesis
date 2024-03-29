import matplotlib.pyplot as plt
from numpy import array
import pandas as pd
from numpy import int64
from sklearn import preprocessing
from sklearn import datasets
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
#import mglearn

### <b> https://stackabuse.com/introduction-to-neural-networks-with-scikit-learn/  </b> #################

# Location of dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=names)

# Assign data from first four columns to X variable
X = irisdata.iloc[:, 0:4]
#print(X)

# Assign data from first fifth columns to y variable
y = irisdata.select_dtypes(include=[object])
y.Class.unique()
array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], dtype=object)

le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Feature Scaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Training and Predictions
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())
predictions = mlp.predict(X_test)

#Evaluating the Algorithm

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
