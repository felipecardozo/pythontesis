import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

file = open("result.txt", "w+")


def perform_data_player(player):
    path = os.getcwd() + "/datasets" + "/output_" + player + ".csv"
    print(path)
    # Read dataset to pandas dataframe
    dotadata = ""
    try:
        dotadata = pd.read_csv(path, sep=',', error_bad_lines=False, index_col=False)

        # Assign data from first four columns to X variable
        X = dotadata.iloc[:, 0:8]
        y = dotadata.iloc[:, 8]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

        # Feature Scaler
        scaler = StandardScaler()
        scaler.fit(X_train)

        # Training and Predictions
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
        mlp.fit(X_train, y_train.values.ravel())
        predictions = mlp.predict(X_test)

        # Evaluating the Algorithm
        confusion = confusion_matrix(y_test, predictions)
        classification = classification_report(y_test, predictions)

        file.write("player " + player)
        file.write("\n")
        file.write("confusion_matrix ============================\n")
        file.write(np.array2string(confusion))
        file.write("\n")

        file.write("classification_report ============================\n")
        file.write(classification)
        file.write("\n")


    except FileNotFoundError:
        print(player + " doesnt exist")


def iterate_players():
    players = ["105248644", "34505203", "72312627", "82262664", "101356886",
               "106573901", "132851371", "134556694", "159020918", "92423451",
               "116585378", "121769650", "184138153", "73562326", "87278757",
               "106863163", "125581247", "94296097", "94738847", "101695162",
               "86745912", "94155156", "111620041", "41231571", "25907144"]
    for player in players:
        perform_data_player(player)

    file.close()


iterate_players()
