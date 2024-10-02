import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pruning_missing_data import impute_by_mice

import numpy as np
import pytest


@pytest.fixture
def init():
    return lambda :np.random.seed(0)


@pytest.fixture
def test_data():
    np.random.seed(0)

    sample_size = 100

    B = np.array([
        [0.00      , 0.00      , 0.00       , 0.88850626, 0.00      , 0.00      ],
        [0.5645509 , 0.00      , 0.37636727 , 0.00      , 0.00      , 0.00      ],
        [0.00      , 0.00      , 0.00       , 0.88850626, 0.00      , 0.00      ],
        [0.00      , 0.00      , 0.00       , 0.00      , 0.00      , 0.00      ],
        [0.9780437 , 0.00      , -0.12225546, 0.00      , 0.00      , 0.00      ],
        [0.88583761, 0.00      , 0.00       , 0.00      , 0.00      , 0.00      ]
    ])
    causal_order = np.array([3, 0, 2, 1, 4, 5])

    scales = [0.2, 0.2, 0.2, 1, 0.2, 0.2]

    e = np.array([np.random.uniform(-np.sqrt(3 * s), np.sqrt(3 * s), size=sample_size) for s in scales])
    X = (np.linalg.pinv(np.eye(len(B)) - B) @ e).T

    prop_missing = 0.1 * np.ones(X.shape[1])

    missing_pos = []
    for i, prop in enumerate(prop_missing):
        mask = np.random.uniform(0, 1, size=len(X))
        missing_pos.append(mask < prop)
    missing_pos = np.array(missing_pos).T

    true_value = X.copy()
    true_value[~missing_pos] = np.nan
    X[missing_pos] = np.nan

    return X, true_value, missing_pos


@pytest.fixture
def test_data_d():
    np.random.seed(0)

    sample_size = 100

    B = np.array([
        [0.00      , 0.00      , 0.00       , 0.88850626, 0.00      , 0.00      ],
        [0.5645509 , 0.00      , 0.37636727 , 0.00      , 0.00      , 0.00      ],
        [0.00      , 0.00      , 0.00       , 0.88850626, 0.00      , 0.00      ],
        [0.00      , 0.00      , 0.00       , 0.00      , 0.00      , 0.00      ],
        [0.9780437 , 0.00      , -0.12225546, 0.00      , 0.00      , 0.00      ],
        [0.88583761, 0.00      , 0.00       , 0.00      , 0.00      , 0.00      ]
    ])
    causal_order = np.array([3, 0, 2, 1, 4, 5])

    scales = [0.2, 0.2, 0.2, 1, 0.2, 0.2]

    e = np.array([np.random.uniform(-np.sqrt(3 * s), np.sqrt(3 * s), size=sample_size) for s in scales])

    disc_index = 3
    X = np.zeros((sample_size, len(B)))
    for co in causal_order:
        if co == disc_index:
            X[:, co] = B[co, :] @ X.T + e[co, :]
            
            r_prob = np.random.uniform(low=0.0, high=1.0, size=sample_size)
            X[:, co] = np.where(r_prob < X[:, co], 1, 0)
        else:
            X[:, co] = B[co, :] @ X.T + e[co, :]
            
    # add missing
    prop_missing = 0.1 * np.ones(X.shape[1])

    missing_pos = []
    for i, prop in enumerate(prop_missing):
        mask = np.random.uniform(0, 1, size=len(X))
        missing_pos.append(mask < prop)
    missing_pos = np.array(missing_pos).T

    true_value = X.copy()
    true_value[~missing_pos] = np.nan
    X[missing_pos] = np.nan

    return X, true_value, missing_pos, disc_index


def test_success(init, test_data, test_data_d):
    init()

    n_imputations = 3
    maxit = 4

    # continous only
    X, true_value, missing_pos = test_data
    imputed_X_list = impute_by_mice(X, n_imputations=n_imputations, maxit=maxit, seed=1)

    assert len(imputed_X_list) == n_imputations
    for imputed_X in imputed_X_list:
        assert imputed_X.shape == (len(X), X.shape[1])

    # same seed
    X, true_value, missing_pos = test_data
    imputed_X_list2 = impute_by_mice(X, n_imputations=n_imputations, maxit=maxit, seed=1)

    for imputed_X, imputed_X2 in zip(imputed_X_list, imputed_X_list2):
        assert np.isclose(imputed_X, imputed_X2).all()

    # other seed
    X, true_value, missing_pos = test_data
    imputed_X_list3 = impute_by_mice(X, n_imputations=n_imputations, maxit=maxit, seed=2)

    for imputed_X, imputed_X3 in zip(imputed_X_list, imputed_X_list3):
        assert not np.isclose(imputed_X, imputed_X3).all()

    # continuous and discrete
    X_d, true_value_d, missing_pos_d, disc_index = test_data_d

    n_imputations2 = 5
    maxit2 = 6
    is_discrete = [False if i != disc_index else True for i in range(X_d.shape[1])]
    imputed_X_list_d = impute_by_mice(X_d, n_imputations=n_imputations2, maxit=maxit2, is_discrete=is_discrete, seed=1)

    assert len(imputed_X_list_d) == n_imputations2
    for imputed_X in imputed_X_list_d:
        assert imputed_X.shape == (len(X_d), X_d.shape[1])

