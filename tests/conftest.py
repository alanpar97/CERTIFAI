import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier

FEATURE_NAMES = [f"feat_{i}" for i in range(6)]
N_FEATURES = 6


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=N_FEATURES,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        random_state=42,
    )
    X_train = pd.DataFrame(X, columns=FEATURE_NAMES)
    y_train = pd.Series(y, name="target")
    return X_train, y_train


@pytest.fixture
def trained_sklearn_model(synthetic_data):
    X_train, y_train = synthetic_data
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def numpy_trained_model(synthetic_data):
    X_train, y_train = synthetic_data
    X_numpy = X_train.to_numpy()
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_numpy, y_train)
    return model
