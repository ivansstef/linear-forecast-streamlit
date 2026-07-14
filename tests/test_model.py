"""Тести для model.py"""
import os
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from joblib import load

# Додаємо корінь проєкту до шляху
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import (
    train_regression_model,
    save_regression_model,
    evaluate_regression_model,
    save_initial_datasets,
)


@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 1))
    y = 3 * X.ravel() + rng.standard_normal(50)
    return X, y


def test_train_returns_linear_regression(simple_data):
    X, y = simple_data
    X_train, y_train = X[:40], y[:40]
    model = train_regression_model(X_train, y_train)
    assert isinstance(model, LinearRegression)


def test_train_model_predicts(simple_data):
    X, y = simple_data
    model = train_regression_model(X[:40], y[:40])
    preds = model.predict(X[40:])
    assert preds.shape == (10,)


def test_save_and_load_model(tmp_path, simple_data):
    X, y = simple_data
    model = train_regression_model(X, y)
    path = str(tmp_path / "model.joblib")
    save_regression_model(model, path)
    loaded = load(path)
    assert isinstance(loaded, LinearRegression)
    np.testing.assert_allclose(model.coef_, loaded.coef_)


def test_evaluate_prints_mse(capsys, simple_data):
    X, y = simple_data
    model = train_regression_model(X[:40], y[:40])
    evaluate_regression_model(model, X[40:], y[40:])
    captured = capsys.readouterr()
    assert "Mean Squared Error:" in captured.out


def test_save_initial_datasets(tmp_path, simple_data, monkeypatch):
    X, y = simple_data
    monkeypatch.chdir(tmp_path)
    save_initial_datasets(X, y)
    X_loaded = load("X.joblib")
    y_loaded = load("y.joblib")
    np.testing.assert_array_equal(X, X_loaded)
    np.testing.assert_array_equal(y, y_loaded)
