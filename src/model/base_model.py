from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models in the project.
    Enforces a consistent API for training, prediction, saving, and loading.
    """

    def __init__(self, model_name: str = "BaseModel", hyperparameters: dict = None):
        self.model_name = model_name
        self.hyperparameters = hyperparameters if hyperparameters is not None else {}

    @abstractmethod
    def train(self, X, y):
        """
        Train the model on the provided features and target.
        Args:
            X: Feature matrix (pd.DataFrame or np.ndarray)
            y: Target vector/series
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Generate predictions for the provided features.
        Args:
            X: Feature matrix (pd.DataFrame or np.ndarray)
        Returns:
            Predictions (np.ndarray or pd.Series)
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the trained model to disk.
        Args:
            path: File path to save the model.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load a trained model from disk.
        Args:
            path: File path to load the model from.
        """
        pass