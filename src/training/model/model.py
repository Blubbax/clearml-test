from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class ModelLogReg:

    def __init__(self, columns):
        
        base_model = LogisticRegression(solver='liblinear')
        self._model = RFE(base_model, n_features_to_select=columns)

    def get_model(_self):
        return _self._model
    

class ModelKNN:

    def __init__(self, columns):
        
        base_model = KNeighborsClassifier(solver='liblinear')
        self._model = RFE(base_model, n_features_to_select=columns)

    def get_model(_self):
        return _self._model   

