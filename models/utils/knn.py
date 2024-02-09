import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
import math

class KNN:
    def __init__(self, eval, max_K: int) -> None:
        assert max_K > 0
        self.max_K = max_K
        self.eval = eval

    def scale(self, set):
        scaler = MinMaxScaler(feature_range=(0, 1))
        set_scaled = pd.DataFrame(scaler.fit_transform(set.drop(columns=['dzien_rozliczenia', 'id', 'week_day'])))
        set_scaled['id'] = set['id']
        set_scaled['week_day'] = set['week_day']
        return pd.DataFrame(set_scaled)
    
    def knn(self, train, test, params):
        y_train = pd.DataFrame({'id': train.id, 'product1': train.product1, 'product2': train.product2})
        x_train = self.scale(train.drop(columns=['product1', 'product2'])).values.tolist()
        x_test = self.scale(test.drop(columns=['product1', 'product2'])).values.tolist()
        
        model = neighbors.KNeighborsRegressor(n_neighbors = params.n_neighbors, algorithm=params.algorithm, metric=params.metric, weights=params.weights, p=params.p, leaf_size=params.leaf_size)
        model.fit(x_train, y_train)
        pred = pd.DataFrame(model.predict(x_test), columns=['id', 'product1', 'product2'])
        pred['id'] = test['id']
        return pred
    
    def tune_parameters(self, train):
        y_train = pd.DataFrame({'id': train.id, 'product1': train.product1, 'product2': train.product2})
        x_train = self.scale(train.drop(columns=['product1', 'product2'])).values.tolist()

        custom_loss = make_scorer(self.eval.custom_loss, greater_is_better=False)
        params = {'n_neighbors': (np.arange(1, min(self.max_K, len(x_train)))), 'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'), 'metric': ('euclidean', 'manhattan', 'chebyshev', 'minkowski'), 'weights': ('uniform', 'distance'), 'p': (1, 2), 'leaf_size': (10, 20, 30, 40, 50)}
        model = neighbors.KNeighborsRegressor()
        model_params = RandomizedSearchCV(model, params, n_iter = 100, cv=8, verbose=2, n_jobs = -1, scoring=custom_loss)
        model_params.fit(x_train, y_train)
        return pd.DataFrame.from_dict([model_params.best_params_])