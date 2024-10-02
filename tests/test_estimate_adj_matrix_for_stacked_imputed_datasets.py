import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pruning_missing_data import estimate_adj_matrix_for_stacked_imputed_datasets

import numpy as np
import pytest


@pytest.fixture
def init():
    return lambda :np.random.seed(0)


@pytest.fixture
def test_data():
    np.random.seed(1)

    sample_size = 30
    n_datasets = 2

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

    X_list = []
    for i in range(n_datasets):
        e = np.array([np.random.uniform(-np.sqrt(3 * s), np.sqrt(3 * s), size=sample_size) for s in scales])
        X = (np.linalg.pinv(np.eye(len(B)) - B) @ e).T
        X_list.append(X)

    return X_list, B, causal_order


def test_success(init, test_data):
    init()

    X_list, B, causal_order = test_data

    n_samples, n_features = X_list[0].shape

    # normal
    estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order)

    # ad_weight_type
    estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, ad_weight_type="1se")

    # weights
    estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, weights=np.ones(n_samples))

    # prior_knowledge
    prior_knowledge = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1],
    ])
    adj = estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, prior_knowledge=prior_knowledge)
    assert np.all(np.isclose(adj[[0, 1], :], 0))

    # seed
    adj = estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, seed=0)
    adj2 = estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, seed=0)
    adj3 = estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, seed=1)
    assert np.all(np.isclose(adj, adj2))

    x, y = np.argwhere(~np.isclose(B, 0)).T
    assert not np.all(np.isclose(adj[x, y],adj3[x, y]))


def test_exception(init, test_data):
    init()

    X_list, B, causal_order = test_data

    n_samples, n_features = X_list[0].shape

    # X_list: 3-dimension
    try:
        estimate_adj_matrix_for_stacked_imputed_datasets(X_list[0], causal_order)
    except ValueError:
        pass
    else:
        raise AssertionError

    # X_list: not empty
    try:
        estimate_adj_matrix_for_stacked_imputed_datasets(np.array([[[],[]]]), causal_order)
    except ValueError:
        pass
    else:
        raise AssertionError

    # causal_order: length
    try:
        estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError

    # causal_order: unique
    try:
        estimate_adj_matrix_for_stacked_imputed_datasets(X_list, [0, 1, 2, 3, 4, 0])
    except ValueError:
        pass
    else:
        raise AssertionError

    # ad_weight_type: min or 1se
    try:
        estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, ad_weight_type="max")
    except ValueError:
        pass
    else:
        raise AssertionError

    # weights: length
    try:
        estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, weights=np.ones(n_samples - 1))
    except ValueError:
        pass
    else:
        raise AssertionError

    # prior knowledge: shape
    try:
        pk = np.ones((n_features - 1, n_features - 1)) * -1
        estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, prior_knowledge=pk)
    except ValueError:
        pass
    else:
        raise AssertionError

    # seed: int
    try:
        estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, seed=1.1)
    except TypeError:
        pass
    else:
        raise AssertionError
