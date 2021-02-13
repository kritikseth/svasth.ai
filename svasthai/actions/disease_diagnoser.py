import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline

from fuzzywuzzy import fuzz, process
import pickle


def diagnose(symptoms):
    df = pd.read_csv('./symptoms_training.csv')

    # Extracting data and target values
    X = df.drop('prognosis', axis=1)
    Y = df.iloc[:, -1]

    # Label encoding
    le = LabelEncoder()
    Y_le = le.fit_transform(Y)

    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y_le, test_size=0.2, random_state=42)

    # Modelling
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(x_train, y_train)

    # Importing model from pickle file

    # splitting input
    symptoms_list = symptoms.split(',')
    symptoms_list = [symptom.strip() for symptom in symptoms_list]
    symptoms_list = [symptom.replace(' ', '_') for symptom in symptoms_list]

    # convert_symptoms_to_likely
    symptoms_list_fromcols = X.columns.to_list()

    symptoms_final = []

    for symptom in symptoms_list:
        if symptom not in symptoms_final:
            if symptom in symptoms_list_fromcols:
                symptoms_final.append(symptom)

        else:
            match = process.extractOne(
                symptom, symptoms_list_fromcols, scorer=fuzz.token_sort_ratio, score_cutoff=80)
            try:
                if symptoms not in symptoms_final:
                    symptoms_final.append(match[0])
            except:
                pass

    # symptom_indexing
    symptoms_list_fromcols = X.columns.to_list()
    symptoms_indexed = np.zeros(132,)

    for symptom in symptoms_list:
        if symptom in X.columns:
            pos = symptoms_list_fromcols.index(symptom)
            symptoms_indexed[pos] = 1

    # predict value
    x = np.array(symptoms_indexed)
    x = x.reshape(1, -1)

    pred = dt.predict(x)

    pred = str(np.squeeze(le.inverse_transform(pred)))

    return pred
