import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from IPython.display import display


class LSTM_model():
    def __init__(self) -> None:
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.param_grid = {
            'epochs': [10, 20, 30],
            'batch_size': [14, 28, 42]
        }

    def train(self, train_x, train_y, epochs=10, batch_size=14):
        features = train_x.columns[2:]
        target_column = 'product1'

        X = train_x[features].values
        y = train_y[target_column].values.reshape(-1, 1)

        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        self.model = Sequential()
        self.model.add(LSTM(units=50, input_shape=(X.shape[1], X.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(X, y, 
                       epochs=epochs, 
                       batch_size=batch_size)
    
    def predict(self, test):
        test_X = test.drop(columns=['id', 'dzien_rozliczenia']).values
        test_X = self.scaler_X.transform(test_X)
        X_test = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])

        predictions_scaled = self.model.predict(X_test)
        predictions = pd.DataFrame(self.scaler_y.inverse_transform(predictions_scaled), columns=['product1'])
        predictions['id'] = test['id']
        predictions['dzien_rozliczenia'] = test['dzien_rozliczenia']

        return predictions
    
    def evaluate(self, val_x, val_y):
        features = val_x.columns[2:]
        target_column = 'product1'

        X_val = val_x[features].values
        y_val = val_y[target_column].values.reshape(-1, 1)  # Reshape for scaler

        X_val = self.scaler_X.transform(X_val)
        y_val = self.scaler_y.transform(y_val)

        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

        predictions_scaled = self.model.predict(X_val)

        predictions = self.scaler_y.inverse_transform(predictions_scaled)

        rmse = mean_squared_error(y_val, predictions, squared=False)
        return rmse
    
    def hyperparameter_tuning(self, train, val, param_grid=None):
        if param_grid is None:
            param_grid = self.param_grid
            
        best_mse = float('inf')
        best_params = None

        for params in ParameterGrid(param_grid):
            self.train(train.drop(columns=['product1']), 
                            pd.DataFrame({'id': train.id, 'product1': train.product1}),
                            epochs=params['epochs'], 
                            batch_size=params['batch_size'])
            
            mse = self.evaluate(val.drop(columns=['product1']), 
                                      pd.DataFrame({'id': val.id, 'product1': val.product1}))

            if mse < best_mse:
                best_mse = mse
                best_params = params
        params_df = pd.DataFrame.from_dict([best_params])
        return params_df