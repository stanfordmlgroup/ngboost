from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split

Tuple4Array = Tuple[np.array, np.array, np.array, np.array]
Tuple5Array = Tuple[np.array, np.array, np.array, np.array, np.array]


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run slow tests")


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getvalue("slow"):
        pytest.skip("need --slow option to run")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: ")


@pytest.fixture(scope="session")
def boston_data() -> Tuple4Array:
    X, Y = load_boston(True)
    return train_test_split(X, Y, test_size=0.2, random_state=23)


@pytest.fixture(scope="session")
def boston_survival_data() -> Tuple5Array:
    X, Y = load_boston(True)
    X_surv_train, X_surv_test, Y_surv_train, Y_surv_test = train_test_split(
        X, Y, test_size=0.2, random_state=14
    )

    # introduce administrative censoring to simulate survival data
    T_surv_train = np.minimum(Y_surv_train, 30)  # time of an event or censoring
    E_surv_train = (
        Y_surv_train > 30
    )  # 1 if T[i] is the time of an event, 0 if it's a time of censoring
    return X_surv_train, X_surv_test, T_surv_train, E_surv_train, Y_surv_test


@pytest.fixture(scope="session")
def breast_cancer_data() -> Tuple4Array:
    X, Y = load_breast_cancer(True)
    return train_test_split(X, Y, test_size=0.2, random_state=12)
