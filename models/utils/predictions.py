import pandas as pd

class Predictions:
    def __init__(self) -> None:
        pass

    def getPredictions(self, name: str, k: int|None=None, isMobile: bool=False):
        if isMobile:
            preds = pd.read_csv('predictions_mobile/'+name+'.csv', index_col=0)
        else:
            preds = pd.read_csv('predictions/'+name+'.csv', index_col=0)
        if k is None:
            return preds
        return preds[preds['k']==k].drop(columns=['k']).reset_index(drop=True)