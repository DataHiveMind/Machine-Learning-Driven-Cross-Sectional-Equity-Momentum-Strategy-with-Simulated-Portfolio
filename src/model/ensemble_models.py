import joblib
from src.model.base_model import BaseModel

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
try:
    from xgboost import XGBRegressor, XGBClassifier
except ImportError:
    XGBRegressor = XGBClassifier = None
try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except ImportError:
    LGBMRegressor = LGBMClassifier = None

class RandomForestModel(BaseModel):
    def __init__(self, model_type="regressor", hyperparameters=None):
        super().__init__("RandomForest", hyperparameters)
        if model_type == "regressor":
            self.model = RandomForestRegressor(**(hyperparameters or {}))
        else:
            self.model = RandomForestClassifier(**(hyperparameters or {}))

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

class XGBoostModel(BaseModel):
    def __init__(self, model_type="regressor", hyperparameters=None):
        super().__init__("XGBoost", hyperparameters)
        if XGBRegressor is None or XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        if model_type == "regressor":
            self.model = XGBRegressor(**(hyperparameters or {}))
        else:
            self.model = XGBClassifier(**(hyperparameters or {}))

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

class LightGBMModel(BaseModel):
    def __init__(self, model_type="regressor", hyperparameters=None):
        super().__init__("LightGBM", hyperparameters)
        if LGBMRegressor is None or LGBMClassifier is None:
            raise ImportError("lightgbm is not installed.")
        if model_type == "regressor":
            self.model = LGBMRegressor(**(hyperparameters or {}))
        else:
            self.model = LGBMClassifier(**(hyperparameters or {}))

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)