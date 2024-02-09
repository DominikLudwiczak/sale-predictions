import pandas as pd
from IPython.display import display

class Parameters:
    def __init__(self) -> None:
        self.knn = pd.read_csv('parameters/knn.csv', index_col=0)
        self.knn_all = pd.read_csv('parameters/knn_all.csv', index_col=0)
        self.rf = pd.read_csv('parameters/rf.csv', index_col=0)
        self.rf_all = pd.read_csv('parameters/rf_all.csv', index_col=0)

        self.knn_mobile = pd.read_csv('parameters_mobile/knn_all.csv', index_col=0)
        self.rf_mobile = pd.read_csv('parameters_mobile/rf_all.csv', index_col=0)

    def get_parameters(self, name: str, k: int, point_id: int|None = None):
        if name=='knn':
            if point_id is None:
                return self.knn_all[self.knn_all['k']==k].iloc[0]
            else:
                return self.knn[(self.knn['k']==k) & (self.knn['id']==point_id)].iloc[0]
        elif name=='rf':
            if point_id is None:
                return self.rf_all[self.rf_all['k']==k].iloc[0]
            else:
                return self.rf[(self.rf['k']==k) & (self.rf['id']==point_id)].iloc[0]
            
    def get_parameters_mobile(self, name: str, k: int, point_id: int|None = None):
        if name=='knn':
            if point_id is None:
                return self.knn_all[self.knn_all['k']==k].iloc[0]
            else:
                return self.knn[(self.knn['k']==k) & (self.knn['id']==point_id)].iloc[0]
        elif name=='rf':
            if point_id is None:
                return self.rf_all[self.rf_all['k']==k].iloc[0]
            else:
                return self.rf[(self.rf['k']==k) & (self.rf['id']==point_id)].iloc[0]