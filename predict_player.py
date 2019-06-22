import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

path_file = 'output_105248644.csv'

# Read dataset to pandas dataframe
dotadata = pd.read_csv(path_file, sep=',', error_bad_lines=False, index_col=False)
#print(dotadata.head())

print(dotadata.dtypes)

# Assign data from first four columns to X variable
X = dotadata.iloc[:, 0:8]
y = dotadata.iloc[:, 8]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# Feature Scaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Training and Predictions
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
mlp.fit(X_train, y_train.values.ravel())
predictions = mlp.predict(X_test)

#Evaluating the Algorithm
confusionmatrix = confusion_matrix(y_test,predictions)
print(confusionmatrix)
total1=sum(sum(confusionmatrix))
accuracy1=(confusionmatrix[0,0]+confusionmatrix[1,1])/total1
print ('Accuracy               : ', accuracy1)

sensitivity1 = confusionmatrix[0,0]/(confusionmatrix[0,0]+confusionmatrix[0,1])
print('Sensitivity            : ', sensitivity1 )

specificity1 = confusionmatrix[1,1]/(confusionmatrix[1,0]+confusionmatrix[1,1])
print('Specificity            : ', specificity1)

print(classification_report(y_test,predictions))