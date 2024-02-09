import pandas as pd

class ReadData:
    def __init__(self) -> None:
        pass

    def read_Hd(self) -> pd.DataFrame:
        hd = pd.read_csv("../datasets/hd.csv")
        return hd.fillna(0)
    
    def read_Hd_Mobile(self) -> pd.DataFrame:
        hd = pd.read_csv("../datasets/hd_mobile.csv")
        hd = hd[hd.product1 > 0]
        return hd.fillna(0)