# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from .symptoms_to_disease import diagnose_disease
from .disease_diagnoser import diagnose

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
from fuzzywuzzy import process

#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []


class ActionGetHospital(Action):

    def name(self) -> Text:
        return "action_search_hospital"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        entities = tracker.latest_message['entities']

        for e in entities:
            if e['entity'] == 'location':
                location = e['value']

            hospitals = []
            hosp = pd.read_csv('./nhosp.csv')

            if type(location) == str:
                location = location.title()

            for row in range(hosp.shape[0]):
                if location in hosp.iloc[row, 1]:
                    hospitals.append([hosp.iloc[row, 0], hosp.iloc[row, 3]])

            for row in range(hosp.shape[0]):
                if location == hosp.iloc[row, 2]:
                    hospitals.append([hosp.iloc[row, 0], hosp.iloc[row, 3]])

            if len(hospitals) == 0:
                hospital = f'No hospital found in {location}'
            elif len(hospitals) == 1:
                hospital = f'Name: {hospitals[0][0]}\nNumber: {hospitals[0][1]}'
            elif len(hospitals) >= 2:
                hospital = f'1.Name: {hospitals[0][0]}\n  Number: {hospitals[0][1]}\n2.Name: {hospitals[1][0]}\n  Number: {hospitals[1][1]}'

        dispatcher.utter_message(text=hospital)

        return []


class ActionGetDisease(Action):

    def name(self) -> Text:
        return "action_search_disease"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        entities = tracker.latest_message['entities']

        symptoms = []

        for entity in entities:
            if entity['entity'] == 'symptoms':
                symptoms.append(entity['value'])

        symptoms = ', '.join(symptoms)

        disease = diagnose(symptoms)

        dispatcher.utter_message(
            text="Based on the given symptoms, you might have " + disease)

        return []

class ActionGetSideEffects(Action):

    def name(self) -> Text:
        return "action_get_sideeffects"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        entities = tracker.latest_message['entities']

        side_effects_m = [['']]
        side_effects = ''
        medicine = ''

        se = pd.read_csv('./sideeffects.csv')
        drugs = se['drug'].tolist()
        for e in entities:
            if e['entity'] == 'medicine':
                medicine = e['value']
            
                side_effects_m = process.extract(medicine, drugs)
                side_effects = se[se['drug']==side_effects_m[0][0]]['sideeffects'].iloc[0]

        if side_effects_m == [['']]:
            side_effects = f'Could not find the side effects of this medicine in my database. Sorry for the inconvenience!'
        else:
            side_effects = f'Side effects of {side_effects_m[0][0]} are: ' + side_effects
        dispatcher.utter_message(text=side_effects)

        return []
