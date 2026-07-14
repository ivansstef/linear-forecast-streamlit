"""Тести для app.py"""
import os
import numpy as np
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app import load_and_predict, _index_of_closest


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    """Тренує модель і зберігає артефакти у тимчасову папку."""
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from model import train_regression_model, save_regression_model, save_initial_datasets

    tmp = tmp_path_factory.mktemp("data")
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X = np.interp(X, (X.min(), X.max()), (-3, 3))
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_regression_model(X_train, y_train)
    model_path = str(tmp / "linear_regression_model.joblib")
    x_path = str(tmp / "X.joblib")
    y_path = str(tmp / "y.joblib")
    save_regression_model(model, model_path)
    from joblib import dump
    dump(X, x_path)
    dump(y, y_path)
    return model_path, X, y


def test_load_and_predict_returns_array(trained_artifacts):
    model_path, X, y = trained_artifacts
    result = load_and_predict([[0.0]], filename=model_path)
    assert result is not None
    assert np.asarray(result).size == 1


def test_load_and_predict_shape(trained_artifacts):
    model_path, X, y = trained_artifacts
    result = load_and_predict([[1.0], [-1.0]], filename=model_path)
    assert np.asarray(result).shape == (2,)


def test_index_of_closest_exact():
    X = np.array([[0.0], [1.0], [2.0], [-1.0]])
    assert _index_of_closest(X, 1.0) == 1


def test_index_of_closest_approximate():
    X = np.array([[0.5], [1.5], [2.5]])
    idx = _index_of_closest(X, 1.4)
    assert idx == 1


def test_index_of_closest_negative():
    X = np.array([[-2.0], [-1.0], [0.0], [1.0]])
    assert _index_of_closest(X, -1.1) == 1
