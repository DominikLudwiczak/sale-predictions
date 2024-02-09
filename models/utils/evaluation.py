import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Evaluation:
    def __init__(self) -> None:
        self.forecasts = pd.read_csv("../datasets/forecast.csv")
        self.forecasts[(self.forecasts.productdictionaryid == 1)]
        self.forecasts = self.forecasts[['pointofsaleid', 'forecastvalue', 'forecastdate']]
        self.forecasts.sort_values(by=['forecastdate', 'pointofsaleid'], inplace=True)
        self.forecasts = self.forecasts.groupby(['forecastdate', 'pointofsaleid'], as_index=False).median()

    def custom_loss(self, y_pred, y_true) -> float:
        penalty_factor = 1.5
        errors = y_true - y_pred
        underestimation_penalty = np.where(errors > 0, penalty_factor * errors, errors * -1)
        return np.mean(underestimation_penalty)

    def RSME(self, y_pred, y_true):
        result = pd.DataFrame()
        for column in y_pred.columns:
            result[column] = pd.Series(mean_squared_error(y_true[column], y_pred[column], squared=False))
        return result
    
    def MAE(self, y_pred, y_true):
        result = pd.DataFrame()
        for column in y_pred.columns:
            errors = y_true[column] - y_pred[column]
            result[column] = pd.Series(mean_absolute_error(y_true[column], y_pred[column]))
        return result
    
    def MAPE(self, y_pred, y_true):
        result = pd.DataFrame()
        for column in y_pred.columns:
            result[column] = pd.Series(np.mean(np.abs((y_true[column] - y_pred[column]) / y_true[column])) * 100)
        return result
    
    def CUSTOM(self, y_pred, y_true):
        result = pd.DataFrame()
        for column in y_pred.columns:
            result[column] = pd.Series(self.custom_loss(y_pred[column], y_true[column]))
        return result
    
    def compare_models(self, test_set):
        test_set = test_set[['id', 'dzien_rozliczenia', 'product1', 'product2', 'product3', 'product4']]
        merged_df = pd.merge(test_set, self.forecasts, left_on=['dzien_rozliczenia', 'id'], right_on=['forecastdate', 'pointofsaleid'], how='inner')
        merged_df.drop(columns=['pointofsaleid', 'forecastdate'], inplace=True)
        
        mae_forecasts = self.MAE(merged_df[['forecastvalue']].rename(columns={'forecastvalue': 'product1'}), merged_df[['product1']])
        rmse_forecasts = self.RSME(merged_df[['forecastvalue']].rename(columns={'forecastvalue': 'product1'}), merged_df[['product1']])
        mape_forecasts = self.MAPE(merged_df[['forecastvalue']].rename(columns={'forecastvalue': 'product1'}), merged_df[['product1']])
        custom_forecasts = self.CUSTOM(merged_df[['forecastvalue']].rename(columns={'forecastvalue': 'product1'}), merged_df[['product1']])
        return (mae_forecasts, rmse_forecasts, mape_forecasts, custom_forecasts)