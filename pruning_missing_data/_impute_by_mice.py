import os
import shutil
import tempfile
import subprocess

import numpy as np
from sklearn.utils import check_array, check_scalar


def impute_by_mice(X, n_imputations=10, maxit=10, is_discrete=None, seed=None):
    """ multiple imputation by mice.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Target data, where ``n_samples`` is the number of samples
        and ``n_features`` is the number of features.
    n_imputations : int, optional (defualt=10)
        The number of multiple imputations.
    maxit : int, optional (default=10)
        The number of iterations.
    is_discrete : array-like of shape (n_features, ), optional (default=None)
        ``is_discrete[i]`` specifies whether the i-th feature is discrete or not.
        All features are considered continuous if ``is_discrete`` is None.
    seed : int, optional (default=None)
        The seed for random numbers.

    Returns
    -------
    imputed_X_list : list, shape [X, ...]
        The list containing imputed X.
    """

    # check arguments
    X = check_array(X, force_all_finite="allow-nan")

    n_imputations = check_scalar(n_imputations, "n_imputations", int)

    maxit = check_scalar(maxit, "maxit", int)

    if is_discrete is not None:
        is_discrete = check_array(is_discrete, dtype=bool, ensure_2d=False)

    if seed is not None:
        seed = check_scalar(seed, "seed", int)

    try:
        # args
        temp_dir = tempfile.mkdtemp()

        path = os.path.join(os.path.dirname(__file__), "impute_by_mice.r")

        args = [f"--temp_dir={temp_dir}"]
        if seed is not None:
            args.append(f"--rs={seed}")

        # write data
        np.savetxt(temp_dir + "/X.csv", X, delimiter=",", fmt="%s")

        # write params
        np.savetxt(temp_dir + "/n_imputations.csv", [n_imputations], delimiter=",", fmt="%s")
        np.savetxt(temp_dir + "/maxit.csv", [maxit], delimiter=",", fmt="%d")
        if is_discrete is not None:
            is_discrete = [str(e).upper() for e in is_discrete]
            np.savetxt(temp_dir + "/is_discrete.csv", [is_discrete], delimiter=",", fmt="%s")

        # run
        ret = subprocess.run(["Rscript", path, *args], capture_output=True)
        if ret.returncode != 0:
            print(ret.stdout.decode())
            print(ret.stderr.decode())

        # retrieve imputed datasets
        imputed_X_list = []
        fnames = np.loadtxt(temp_dir + "/result_filenames.csv", delimiter=",", skiprows=1, dtype=str)
        for fname in fnames:
            imputed_X = np.loadtxt(temp_dir + "/" + fname, delimiter=",", skiprows=1)
            imputed_X_list.append(imputed_X)
    except FileNotFoundError:
        raise RuntimeError("Rscript is not found.")
    except BaseException as e:
        raise RuntimeError(str(e))
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return imputed_X_list
