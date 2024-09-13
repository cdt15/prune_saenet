import os
import shutil
import tempfile
import subprocess

import numpy as np
from sklearn.utils import check_array, check_scalar


def estimate_adj_matrix_for_stacked_imputed_datasets(X_list, causal_order, is_discrete=None, ad_weight_type="min", weights=None, prior_knowledge=None, seed=None):
    """ estimation of pruned adjacency matrices assuming a common pattern

    Parameters
    ----------
    X_list : list, shape [X, ...]
        Multiple datasets for training, where ``X`` is an dataset.
        
        The shape of ''X'' is (n_samples, n_features),
        where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
    causal_order : list, length = n_features
        The causal order of X_list.
    is_discrete : array-like of shape (n_features, ), optional (default=None)
        ``is_discrete[i]`` specifies whether the i-th feature is discrete or not.
    ad_weight_type : string, "min" or "1se", optional (default="min")
        The name of the method to decide the adaptive weights.
    weights : array-like of shape (n_samples, ), optional (default=None)
        The proportion of the missing data for each row in X. 
    prior_knowledge : array-like, shape (n_features, n_features), optional (default=None)
        The prior knowledge used for the causal discovery, where ``n_features`` is the number of features.

        The elements of prior knowledge matrix are defined as follows [1]_:

        * ``0`` : :math:`x_i` does not have a directed path to :math:`x_j`
        * ``1`` : :math:`x_i` has a directed path to :math:`x_j`
        * ``-1`` : No prior knowledge is available to know if either of the two cases above (0 or 1) is true.
    seed : int, optional (default=None)
        The seed for random numbers.
    
    Returns
    -------
    adjacency_matrix : array-like
        The shape of ``adjacency_matrix`` is (n_features, n_features), where
        ``n_features`` is the number of features.
    """
    # check arguments
    X_list = check_array(X_list, allow_nd=True)
    if len(X_list.shape) != 3:
        raise ValueError("X_list must be 3-dimensional.")
    if 0 in X_list.shape:
        raise ValueError("The shape of X_list mustn't contains 0.")
    
    n_imputations, n_samples, n_features = X_list.shape
    
    causal_order = check_array(causal_order, ensure_2d=False)
    if len(causal_order) != n_features:
        raise ValueError("The length of causal_order must equal to n_features.")
    if not np.all(np.sort(causal_order) == np.arange(n_features)):
        raise ValueError("Elements of causal_order must be unique between 0 and n_features.")

    if is_discrete is not None:
        is_discrete = check_array(is_discrete, dtype=bool, ensure_2d=False)
        if len(is_discrete) != n_features:
            raise ValueError("The length of is_discrete must be equal to n_features.")

        is_discrete = list(map(lambda s: str(s).upper(), is_discrete))
        
    ad_weight_type = check_scalar(ad_weight_type, "ad_weight_type", str)
    if not (ad_weight_type == "min" or ad_weight_type == "1se"):
        raise ValueError("ad_weight_type must be \"min\" or \"1se\".")
    
    if weights is not None:
        weights = check_array(weights, ensure_2d=False)
        if weights.shape != (n_samples, ):
            raise ValueError("The length of weights must be equal to n_samples.")
    else:
        weights = np.ones(n_samples)
    
    if prior_knowledge is not None:
        prior_knowledge = check_array(prior_knowledge)
        if prior_knowledge.shape != (n_features, n_features):
            raise ValueError("The shape of prior_knowledge must be (n_features, n_features).")
    
    if seed is not None:
        seed = check_scalar(seed, "seed", int)
    
    try:
        temp_dir = tempfile.mkdtemp()
        
        path = os.path.join(os.path.dirname(__file__), "estimate_adj_matrix_for_stacked_imputed_datasets.r")
        
        args = [f"--temp_dir={temp_dir}"]
        if seed is not None:
            args.append(f"--rs={seed}")
        
        # write data
        X_names = []
        for i, X in enumerate(X_list):
            X_name = f"X_list_{i:d}.csv"
            np.savetxt(temp_dir + "/" + X_name, X, delimiter=",")
            X_names.append([X_name])
        
        # write params
        np.savetxt(temp_dir + "/X_names.csv", X_names, delimiter=",", fmt="%s")
        np.savetxt(temp_dir + "/causal_order.csv", causal_order, delimiter=",", fmt="%d")
        if is_discrete is not None:
            np.savetxt(temp_dir + "/is_discrete.csv", [is_discrete], delimiter=",", fmt="%s")
        np.savetxt(temp_dir + "/ad_weight_type.csv", [ad_weight_type], delimiter=",", fmt="%s")
        np.savetxt(temp_dir + "/weights.csv", weights, delimiter=",")
        if prior_knowledge is not None:
            np.savetxt(temp_dir + "/prior_knowledge.csv", prior_knowledge, delimiter=",")
        
        # run
        ret = subprocess.run(["Rscript", path, *args], capture_output=True)
        if ret.returncode != 0:
            print(ret.stdout.decode())
            print(ret.stderr.decode())

        # retrieve result
        adjacency_matrix = np.loadtxt(temp_dir + "/result_adj_mat.csv", delimiter=",", skiprows=1)
    except FileNotFoundError as e:
        raise RuntimeError("Rscript is not found.")
    except BaseException as e:
        raise RuntimeError(str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    return adjacency_matrix
