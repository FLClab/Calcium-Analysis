from scipy.spatial import distance_matrix
from scipy import sparse
from scipy.optimize import milp, LinearConstraint
import numpy as np

def _validate_distribution(values, weights):
    """
    Validate the values and weights from a distribution input of `cdf_distance`
    and return them as ndarray objects.

    Parameters
    ----------
    values : array_like
        Values observed in the (empirical) distribution.
    weights : array_like
        Weight for each value.

    Returns
    -------
    values : ndarray
        Values as ndarray.
    weights : ndarray
        Weights as ndarray.

    """
    # Validate the value array.
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        raise ValueError("Distribution can't be empty.")

    # Validate the weight array, if specified.
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if len(weights) != len(values):
            raise ValueError('Value and weight array-likes for the same '
                             'empirical distribution must be of the same size.')
        if np.any(weights < 0):
            raise ValueError('All weights must be non-negative.')
        if not 0 < np.sum(weights) < np.inf:
            raise ValueError('Weight array-like sum must be positive and '
                             'finite. Set as None for an equal distribution of '
                             'weight.')

        return values, weights

    return values, None

def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
                                       deltas)), 1/p)

def wasserstein_distance_nd(u_values, v_values, u_weights=None, v_weights=None):
    
    m, n = len(u_values), len(v_values)
    u_values = np.asarray(u_values)
    v_values = np.asarray(v_values)

    if u_values.ndim > 2 or v_values.ndim > 2:
        raise ValueError('Invalid input values. The inputs must have either '
                         'one or two dimensions.')
    # if dimensions are not equal throw error
    if u_values.ndim != v_values.ndim:
        raise ValueError('Invalid input values. Dimensions of inputs must be '
                         'equal.')
    # if data is 1D then call the cdf_distance function
    if u_values.ndim == 1 and v_values.ndim == 1:
        return _cdf_distance(1, u_values, v_values, u_weights, v_weights)

    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)
    # if number of columns is not equal throw error
    if u_values.shape[1] != v_values.shape[1]:
        raise ValueError('Invalid input values. If two-dimensional, '
                         '`u_values` and `v_values` must have the same '
                         'number of columns.')

    # if data contains np.inf then return inf or nan
    if np.any(np.isinf(u_values)) ^ np.any(np.isinf(v_values)):
        return np.inf
    elif np.any(np.isinf(u_values)) and np.any(np.isinf(v_values)):
        return np.nan

    # create constraints
    A_upper_part = sparse.block_diag((np.ones((1, n)), ) * m)
    A_lower_part = sparse.hstack((sparse.eye(n), ) * m)
    # sparse constraint matrix of size (m + n)*(m * n)
    A = sparse.vstack((A_upper_part, A_lower_part))
    A = sparse.coo_array(A)

    # get cost matrix
    D = distance_matrix(u_values, v_values, p=2)
    cost = D.ravel()

    # create the minimization target
    p_u = np.full(m, 1/m) if u_weights is None else u_weights/np.sum(u_weights)
    p_v = np.full(n, 1/n) if v_weights is None else v_weights/np.sum(v_weights)
    b = np.concatenate((p_u, p_v), axis=0)

    # solving LP
    constraints = LinearConstraint(A=A.T, ub=cost)
    opt_res = milp(c=-b, constraints=constraints, bounds=(-np.inf, np.inf))
    return -opt_res.fun