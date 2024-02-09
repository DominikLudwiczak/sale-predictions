import pandas as pd
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

class RandomForest:
    def __init__(self, eval) -> None:
        self.eval = eval

    def randForest(self, train, test, params):
        train_x = train.drop(columns=['dzien_rozliczenia', 'product1', 'product2'])
        test_x = test.drop(columns=['dzien_rozliczenia', 'product1', 'product2'])
        train_y = train[['product1', 'product2']].values
        rf = RandomForestRegressor(criterion=params.criterion, max_depth=int(params.max_depth) if not math.isnan(params.max_depth) else None, min_samples_split=params.min_samples_split, min_samples_leaf=params.min_samples_leaf, max_features=params.max_features, n_estimators = params.n_estimators, random_state = 23)
        rf.fit(train_x, train_y)
        pred = pd.DataFrame(rf.predict(test_x), columns=['product1', 'product2'])
        pred['id'] = test.id
        return pred

    def tune_parameters(self, train):
        train_x = train.drop(columns=['dzien_rozliczenia', 'product1', 'product2'])
        train_y = train[['product1', 'product2']].values

        custom_loss = make_scorer(self.eval.custom_loss, greater_is_better=False)
        params = {'criterion': ("squared_error", "absolute_error", "poisson", "friedman_mse"), 'max_depth': (None, 5, 10, 15, 20), 'min_samples_split': (2, 5, 10, 15, 20), 'min_samples_leaf': (1, 2, 5, 10, 15, 20), 'max_features': ("sqrt", "log2"), 'n_estimators': (10, 50, 100, 200, 500)}
        model = RandomForestRegressor()
        rf = RandomizedSearchCV(model, params, n_iter = 100, cv=8, verbose=2, n_jobs = -1, scoring=custom_loss)
        rf.fit(train_x, train_y)
        return pd.DataFrame.from_dict([rf.best_params_])