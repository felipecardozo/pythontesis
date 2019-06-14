import pandas as pd
from numpy import array
from numpy import vstack
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

path_file = 'data_set.csv'
names = ['accountId', 'assists', 'heroId', 'heroKills', 'lane', 'neutralKills', 'gameMode', 'totalXp', 'totalGold',
         'win']

# Read dataset to pandas dataframe
# dotadata = pd.read_csv(path_file, names=names)
#dotadata = pd.read_csv(path_file, sep=',', error_bad_lines=False, index_col=False, dtype='object', names=names)
dotadata = pd.read_csv(path_file, sep=',', error_bad_lines=False, index_col=False)
#print(dotadata.head())

print(dotadata.dtypes)

# Assign data from first four columns to X variable
X = dotadata.iloc[:, 0:9]
y = dotadata.iloc[:, 9]

# Assign data from first fifth columns to y variable
#y = dotadata.select_dtypes(include=[object])
#y.win.unique()
#array(['Win', 'NoWin'], dtype=object)

#le = preprocessing.LabelEncoder()

#y = y.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#print(X_train['accountId'].dtype)
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

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))