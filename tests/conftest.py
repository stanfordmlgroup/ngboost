from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split

Tuple4Array = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Tuple5Array = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run slow tests")


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getvalue("slow"):
        pytest.skip("need --slow option to run")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: ")


@pytest.fixture(scope="session", autouse=True)
def set_seed():
    np.random.seed(0)


@pytest.fixture(scope="session")
def california_housing_data() -> Tuple4Array:
    X, Y = fetch_california_housing(return_X_y=True)
    return train_test_split(X, Y, test_size=0.2, random_state=23)


@pytest.fixture(scope="session")
def california_housing_survival_data() -> Tuple5Array:
    X, Y = fetch_california_housing(return_X_y=True)
    X_surv_train, X_surv_test, Y_surv_train, Y_surv_test = train_test_split(
        X, Y, test_size=0.2, random_state=14
    )

    # calculate threshold for censoring data
    censor_threshold = np.quantile(Y_surv_train, 0.75)
    # introduce administrative censoring to simulate survival data
    T_surv_train = np.minimum(
        Y_surv_train, censor_threshold
    )  # time of an event or censoring
    E_surv_train = (
        Y_surv_train > censor_threshold
    )  # 1 if T[i] is the time of an event, 0 if it's a time of censoring
    return X_surv_train, X_surv_test, T_surv_train, E_surv_train, Y_surv_test


@pytest.fixture(scope="session")
def breast_cancer_data() -> Tuple4Array:
    X, Y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, Y, test_size=0.2, random_state=12)
