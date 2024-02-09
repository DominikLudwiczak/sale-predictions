import pandas as pd
import numpy as np
from IPython.display import display

class SimplePredictors:
    def __init__(self, df) -> None:
        self.df = df

    def avg(self, prediction_date, last_n_days, id) -> float:
        if len(self.df[self.df.id == id]) < last_n_days:
            raise Exception("Insufficient data for prediction")
        else:
            sales_data = self.df[(self.df.dzien_rozliczenia < prediction_date) & (self.df.id == id)].iloc[:last_n_days]
            sales_data.drop(columns=['id', 'dzien_rozliczenia'], inplace=True)
            pred = pd.DataFrame(sales_data.mean(axis=0)).T
            pred['id'] = id
            pred['dzien_rozliczenia'] = prediction_date
            return pred.fillna(0)
        
    def back_n_days(self, prediction_date, n_days, id) -> float:
        pred_date = pd.to_datetime(prediction_date)
        sales_data = pd.DataFrame(self.df[(self.df.dzien_rozliczenia == str((pred_date - pd.DateOffset(days=n_days)).date())) & (self.df.id == id)])
        if len(sales_data) == 0:
            sales_data = pd.DataFrame(columns=self.df.columns)
            sales_data.loc[0] = [0 for i in range(len(self.df.columns))]
        sales_data['dzien_rozliczenia'] = prediction_date
        sales_data['id'] = id
        return sales_data.fillna(0)
        
    def same_days_last_n_weeks(self, prediction_date, n_weeks, id):
        prediction_date = pd.to_datetime(prediction_date)
        sales_data = pd.DataFrame()
        for i in range(n_weeks):
            sales_data = pd.concat([sales_data, self.df[(self.df.dzien_rozliczenia == str((prediction_date - pd.DateOffset(weeks=i+1)).date())) & (self.df.id == id)]])
        if len(sales_data) == 0:
            sales_data = pd.DataFrame(columns=self.df.columns)
            sales_data.loc[0] = [0 for i in range(len(self.df.columns))]
        return sales_data

    def same_days_last_n_weeks_avg(self, prediction_date, n_weeks, id) -> float:
        sales_data = self.same_days_last_n_weeks(prediction_date, n_weeks, id)
        sales_data.drop(columns=['id', 'dzien_rozliczenia'], inplace=True)
        pred = pd.DataFrame(sales_data.mean(axis=0)).T
        pred['id'] = id
        pred['dzien_rozliczenia'] = prediction_date
        return pred.fillna(0)

    def EWMA(self, prediction_date, n_weeks, id) -> float:
        sales_data = self.same_days_last_n_weeks(prediction_date, n_weeks, id)
        if sales_data.iloc[0].dzien_rozliczenia == 0:
            pred = sales_data
        else:
            sales_data.drop(columns=['id', 'dzien_rozliczenia'], inplace=True)
            sales_data = sales_data.ewm(span=n_weeks).mean()
            pred = pd.DataFrame(sales_data.mean(axis=0)).T
            pred.columns = sales_data.columns
        pred['id'] = id
        pred['dzien_rozliczenia'] = prediction_date
        return pred.fillna(0)
    
    def mobile_last_n_times(self, prediction_date, n_times, id) -> float:
        if len(self.df[self.df.id == id]) < n_times:
            raise Exception("Insufficient data for prediction")
        else:
            sales_data = self.df[(self.df.dzien_rozliczenia < prediction_date) & (self.df.id == id)].iloc[:n_times]
            sales_data.drop(columns=['id', 'dzien_rozliczenia'], inplace=True)
            pred = pd.DataFrame(sales_data.mean(axis=0)).T
            pred['id'] = id
            pred['dzien_rozliczenia'] = prediction_date
            return pred.fillna(0)
