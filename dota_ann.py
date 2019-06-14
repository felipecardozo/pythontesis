import pandas as pd
from numpy import array
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

path_file='data_set.csv'
names = ['accountId','assists','heroId','heroKills','lane','neutralKills','gameMode','totalXp','totalGold','win']

# Read dataset to pandas dataframe
#dotadata = pd.read_csv(path_file, names=names)
dotadata = pd.read_csv(path_file, sep=',', error_bad_lines=False, index_col=False, dtype='unicode', names=names)

# Assign data from first four columns to X variable
X = dotadata.iloc[:, 0:9]

# Assign data from first fifth columns to y variable
y = dotadata.select_dtypes(include=[object])
y.win.unique()
array(['Win', 'NoWin'], dtype=object)

le = preprocessing.LabelEncoder()

y = y.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Feature Scaler
scaler = StandardScaler()
scaler.fit(X_train)