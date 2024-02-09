import pandas as pd
import numpy as np
import itertools
from IPython.display import display

class RollingCV:
    def __init__(self, df, K: int, test_duration_days: int, validation_duration_days: int, get_n_days_back_for_ML: int) -> None:
        assert K > 0
        assert test_duration_days > 0
        assert get_n_days_back_for_ML > 0

        self.test_duration_days = test_duration_days
        self.validation_duration_days = validation_duration_days
        self.get_n_days_back_for_ML = get_n_days_back_for_ML

        weather = pd.read_csv('../datasets/forecastweather.csv')
        weather['forecastdate'] = weather.forecastdate.astype(str).str.slice(stop=10)
        weather = weather.groupby(['pointofsaleid', 'forecastdate']).mean(numeric_only=True)
        weather.drop(columns=['idforecastweather', 'forecastingday'], inplace=True)

        self.K = K
        self.sets = []
        self.MLsets = []

        df.sort_values(by=['dzien_rozliczenia'], ascending=True, inplace=True)

        dates = df['dzien_rozliczenia'].unique()
        dates = pd.date_range(start=np.min(dates), end=np.max(dates)).astype(str).str.slice(stop=10)
        pointsofsale = df['id'].unique()

        products_to_predict = ['product1', 'product2']

        data = itertools.product(dates, pointsofsale)
        df_ML = pd.DataFrame(data, columns=['dzien_rozliczenia', 'id'])
        for product in products_to_predict:
            df_ML[product] = df_ML.apply(
                lambda row: df[product][(df['dzien_rozliczenia'] == row.dzien_rozliczenia) & (df['id'] == row.id)].values[0]
                            if row.dzien_rozliczenia in df['dzien_rozliczenia'][(df['id'] == row.id)].unique() else 0.0
            , axis=1)
        df_ML['temperature'] = df_ML.apply(
            lambda row: weather['temperature'][(weather.index.get_level_values('forecastdate') == row.dzien_rozliczenia) & (weather.index.get_level_values('pointofsaleid') == row.id)].values[0]
                        if row.dzien_rozliczenia in weather.index.get_level_values('forecastdate')[(weather.index.get_level_values('pointofsaleid') == row.id)].unique() else 0.0
        , axis=1)
        df_ML['snow'] = df_ML.apply(
            lambda row: weather['snow'][(weather.index.get_level_values('forecastdate') == row.dzien_rozliczenia) & (weather.index.get_level_values('pointofsaleid') == row.id)].values[0]
                        if row.dzien_rozliczenia in weather.index.get_level_values('forecastdate')[(weather.index.get_level_values('pointofsaleid') == row.id)].unique() else 0.0
        , axis=1)
        df_ML['rain'] = df_ML.apply(
            lambda row: weather['rain'][(weather.index.get_level_values('forecastdate') == row.dzien_rozliczenia) & (weather.index.get_level_values('pointofsaleid') == row.id)].values[0]
                        if row.dzien_rozliczenia in weather.index.get_level_values('forecastdate')[(weather.index.get_level_values('pointofsaleid') == row.id)].unique() else 0.0
        , axis=1)
        
        df_ML['week_day'] = pd.to_datetime(df_ML.dzien_rozliczenia).dt.dayofweek
        for product in products_to_predict:
            for i in range(1, get_n_days_back_for_ML+1):
                df_ML[f'-{i}_{product}'] = df_ML.groupby('id')[product].shift(i).fillna(0)
        df_ML.sort_values(by=['dzien_rozliczenia'], ascending=True, inplace=True)
        df_ML = pd.merge(df_ML, df.drop(columns=['product1', 'product2', 'product3', 'product4']), on=['dzien_rozliczenia', 'id'], how='inner')
        df['dzien_rozliczenia'] = pd.to_datetime(df.dzien_rozliczenia)
        first_date = pd.to_datetime(df.dzien_rozliczenia.min()) + pd.Timedelta(days=get_n_days_back_for_ML)
        df_ML = df_ML[df_ML.dzien_rozliczenia >= first_date.strftime('%Y-%m-%d')]
        last_date = pd.to_datetime(df.dzien_rozliczenia.max())
        df['dzien_rozliczenia'] = df.dzien_rozliczenia.astype(str)

        train_n = self.test_duration_days + self.validation_duration_days
        val_n = self.test_duration_days

        for k in range(0, K):
            n = k*(test_duration_days + validation_duration_days)
            treshold_date = last_date - pd.Timedelta(days=n)
            
            train_until = pd.to_datetime(treshold_date - pd.Timedelta(days=train_n)).strftime('%Y-%m-%d')
            training_ML = df_ML[df_ML.dzien_rozliczenia <= train_until]
            
            val_until = pd.to_datetime(treshold_date - pd.Timedelta(days=val_n)).strftime('%Y-%m-%d')
            val_ML = df_ML[(df_ML.dzien_rozliczenia > train_until) & (df_ML.dzien_rozliczenia <= val_until)]

            test_ML = df_ML[(df_ML.dzien_rozliczenia > val_until) & (df_ML.dzien_rozliczenia <= pd.to_datetime(treshold_date).strftime('%Y-%m-%d'))]

            training_ML.reset_index(drop=True, inplace=True)
            val_ML.reset_index(drop=True, inplace=True)
            test_ML.reset_index(drop=True, inplace=True)
            self.MLsets.insert(0, [training_ML, val_ML, test_ML])

            training = df[df.dzien_rozliczenia.isin(training_ML.dzien_rozliczenia)]
            val = df[df.dzien_rozliczenia.isin(val_ML.dzien_rozliczenia)]
            test = df[df.dzien_rozliczenia.isin(test_ML.dzien_rozliczenia)]
            training.reset_index(drop=True, inplace=True)
            val.reset_index(drop=True, inplace=True)
            test.reset_index(drop=True, inplace=True)
            self.sets.insert(0, [training, val, test])

    def getTestSets(self, isML: bool = False, id=None):
        if isML:
            if id is not None:
                return [x[2][x[2].id == id].reset_index(drop=True) for x in self.MLsets]
            return [x[2].reset_index(drop=True) for x in self.MLsets]
        if id is not None:
            return [x[2][x[2].id == id].reset_index(drop=True) for x in self.sets]
        return [x[2].reset_index(drop=True) for x in self.sets]

    def getKthTrainSet(self, k: int, isML: bool = False, id=None, isRegression: bool = False):
        KthSet = self.getKthSet(k, isML)[0]
        if isRegression:
            KthSet = KthSet.loc[:, ~KthSet.columns.str.contains('-')]
        if id is not None:
            result = KthSet[KthSet.id == id].reset_index(drop=True)
            if isML:
                valSet = self.getKthValSet(k, isML, id, isRegression)
                result = pd.concat([result, valSet], ignore_index=True)
            return result
        result = KthSet.reset_index(drop=True)
        if isML:
            valSet = self.getKthValSet(k=k, isML=isML, isRegression=isRegression)
            result = pd.concat([result, valSet], ignore_index=True)
        return result
        
    def getKthTestSet(self, k: int, isML: bool = False, id=None, isRegression: bool = False):
        KthSet = self.getKthSet(k, isML)[2]
        if isRegression:
            KthSet = KthSet.loc[:, ~KthSet.columns.str.contains('-')]
        if id is not None:
            return KthSet[KthSet.id == id].reset_index(drop=True)
        return KthSet.reset_index(drop=True)
    
    def getKthValSet(self, k: int, isML: bool = False, id=None, isRegression: bool = False):
        KthSet = self.getKthSet(k, isML)[1]
        if isRegression:
            KthSet = KthSet.loc[:, ~KthSet.columns.str.contains('-')]
        if id is not None:
            return KthSet[KthSet.id == id].reset_index(drop=True)
        return KthSet.reset_index(drop=True)
    
    def getKthSet(self, k: int, isML: bool):
        if not (0 < k <= self.K):
            raise Exception("k must be greater or equal 1 and less or equal K")
        if isML:
            return self.MLsets[k-1]
        return self.sets[k-1]
