from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, EasterMonday, Easter
from pandas.tseries.offsets import Day, CustomBusinessDay
import json
import pickle


class FrenchJoursFeries(AbstractHolidayCalendar):
    """ Custom Holiday calendar for France based on
        https://en.wikipedia.org/wiki/Public_holidays_in_France
      - 1 January: New Year's Day
      - Moveable: Easter Monday (Monday after Easter Sunday)
      - 1 May: Labour Day
      - 8 May: Victory in Europe Day
      - Moveable Ascension Day (Thursday, 39 days after Easter Sunday)
      - 14 July: Bastille Day
      - 15 August: Assumption of Mary to Heaven
      - 1 November: All Saints' Day
      - 11 November: Armistice Day
      - 25 December: Christmas Day
    """
    rules = [
        Holiday('New Years Day', month=1, day=1),
        EasterMonday,
        Holiday('Labour Day', month=5, day=1),
        Holiday('Victory in Europe Day', month=5, day=8),
        Holiday('Ascension Day', month=1, day=1, offset=[Easter(), Day(39)]),
        Holiday('Bastille Day', month=7, day=14),
        Holiday('Assumption of Mary to Heaven', month=8, day=15),
        Holiday('All Saints Day', month=11, day=1),
        Holiday('Armistice Day', month=11, day=11),
        Holiday('Christmas Day', month=12, day=25)
    ]

def _create_features(df, holidays, label=None):#input/output features pour algo
        """
        Creates time series features from datetime index
        """
        df['date'] = df.index
        df['heure'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['jour_semaine'] = df['date'].dt.dayofweek
        df['trimestre'] = df['date'].dt.quarter
        df['mois'] = df['date'].dt.month
        df['annee'] = df['date'].dt.year
        df['jour_annee'] = df['date'].dt.dayofyear
        df['jour_mois'] = df['date'].dt.day
        df['semaine'] = df['date'].dt.week
        seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
        month_to_season = dict(zip(range(1,13), seasons))
        df['saison'] = df['date'].dt.month.map(month_to_season)
        df['jour_ferie'] = df.index.isin(holidays)
        df['jour_ferie'] = df['jour_ferie'].astype(int)
        bins = [0,6,11,13,19,23]#0-6h nuit/7-12h matin/12-14h pause dejeuner/14-20h apres midi/20-24h soir 
        labels = [0,1,2,3,4]#nuit/matin/pausedejeuner/apresmidi/soir
        df['plage_horaire'] = pd.cut(df['heure'], bins=bins, labels=labels, include_lowest=True)#to include 0
        df['plage_horaire'] = df['plage_horaire'].astype(int)

        df = df.loc[~((df['heure'] < 8) | (df['heure'] > 19))]


        X = df[['heure', 'jour_semaine', 'minute','trimestre','mois','annee','jour_annee', 'jour_mois', 'semaine','saison', 'jour_ferie', 'plage_horaire']]
        if label:
            y = df[label]
            return X, y
        return X

def _preprocess_data():

    f = open("./pmr_data.pkl", "rb")
    
    pmr_json_data = pickle.load(f)

    df = pd.DataFrame(pmr_json_data['values'], index =pmr_json_data['index'], 
                                              columns =['slotStatus'])

    df.index = pd.to_datetime(df.index)
    
    # remove duplicate indices
    df = df[~df.index.duplicated()]

    df = df.asfreq(('15min')) #add missing dates to df via hour filter

    df = df.fillna(method='ffill')#replace nan with preceding value (forward fill method)

    cal = FrenchJoursFeries()

    holidays = cal.holidays(start=df.index.min(), end=df.index.max())

    #print(df.isnull().any())

    X, y = _create_features(df, holidays, label='slotStatus') 

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    np.save('x_train.npy', x_train)
    np.save('x_test.npy', x_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
     
if __name__ == '__main__':

    print('Preprocessing data...')
    _preprocess_data()
