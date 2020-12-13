import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

from fuzzywuzzy import fuzz, process
import pickle


# Function to predict value for single entered row of values(in the form of a list or array)
def predict_value(x):
    x = np.array(x)
    x = x.reshape(1, -1)

    pred = dt.predict(x)

    pred = str(np.squeeze(le.inverse_transform(pred)))

    return pred


def input_split(symptoms_input):
    symptoms_list = symptoms_input.split(',')
    symptoms_list = [symptom.strip() for symptom in symptoms_list]
    symptoms_list = [symptom.replace(' ', '_') for symptom in symptoms_list]

    return symptoms_list


def convert_symptoms_to_likely(symptoms_list):

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

    return symptoms_final


def symptom_indexing(symptoms_list):
    symptoms_list_fromcols = X.columns.to_list()
    symptoms_indexed = np.zeros(132,)

    for symptom in symptoms_list:
        if symptom in X.columns:
            pos = symptoms_list_fromcols.index(symptom)
            symptoms_indexed[pos] = 1

    return symptoms_indexed


def diagnose_disease(symptoms):
    symptoms_list = input_split(symptoms)
    symptoms_final = convert_symptoms_to_likely(symptoms_list)
    symptoms_indexed = symptom_indexing(symptoms_final)
    pred_disease = predict_value(symptoms_indexed)

    return pred_disease


if __name__ == "__main__":

    df = pd.read_csv('random_seed\siemens-healthcare\symptoms_training.csv')

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

    # pickle.dump(dt, open('model.sav', 'wb'))

    # y_pred = dt.predict(x_test)
    # print("Accuracy is %.2f%%" % (accuracy_score(y_test, y_pred) * 100))

    # Testing using a predefined set of symptoms
    symptoms = "throat irritation, loss of smell, cough"
    # symptoms = input("Enter comma separated symptoms")

    disease = diagnose_disease(symptoms)

    print(f"Based on the given symptoms, you might have {disease}")
