import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from IPython.display import display
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

class CNN:
    def __init__(self):
        self.model = None
        self.param_grid = {
            'epochs': [10, 20, 30],
            'batch_size': [14, 28, 42]
        }
    
    def train(self, train_x, train_y, epochs=10, batch_size=28):
        features = train_x.columns[2:]
        target_column = 'product1'

        X = train_x[features].values
        y = train_y[target_column].values.reshape(-1, 1) 
        X = X.reshape(X.shape[0], X.shape[1], 1)
        self.model = Sequential()

        self.model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])))

        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Flatten())

        self.model.add(Dense(64, activation='relu'))

        self.model.add(Dense(1, activation='linear'))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.model.fit(X, y, 
                       epochs=epochs, 
                       batch_size=batch_size)
        
    def predict(self, test):
        features = test.columns[2:]
        X = test[features].values
        X = X.reshape(X.shape[0], X.shape[1], 1)
        y_pred = pd.DataFrame(self.model.predict(X), columns=['product1'])
        y_pred['id'] = test['id']
        y_pred['dzien_rozliczenia'] = test['dzien_rozliczenia']
        return y_pred
    
    def evaluate(self, val_x, val_y):
        features = val_x.columns[2:]
        X_val = val_x[features].values
        y_val = val_y['product1'].values.reshape(-1, 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        y_pred = self.model.predict(X_val)

        return np.sqrt(mean_squared_error(y_val, y_pred))
    
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