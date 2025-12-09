import os
import numpy as np
from itertools import combinations
from scipy.special import expit
from scipy.stats import beta
import numpy as np

# ----------------------------------------------------------------------
# Math & Transformation Utilities
# ----------------------------------------------------------------------

def robust_sigmoid(x):
    """Numerically stable sigmoid function."""
    return expit(x)

def normalize_weights(w):
    """Ensure weights sum to 1 and are non-negative."""
    w = np.maximum(w, 1e-10)
    return w / np.sum(w)

def safe_inverse(M, jitter=1e-6):
    """Robust matrix inversion with fallback to pseudo-inverse."""
    try:
        return np.linalg.inv(M + np.eye(M.shape[0]) * jitter)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def dirichlet_transform(u, alpha):
    N_pars = len(u)
    u = u[:-1]
    """
    Vectorized Dirichlet prior transform via stick-breaking.
    
    Parameters
    ----------
    u : array-like, shape (K-1,)
        Values in (0,1) from dynesty's unit cube.
    alpha : array-like, shape (K,)
        Dirichlet concentration parameters.
    
    Returns
    -------
    x : ndarray, shape (K,)
        A point on the simplex drawn from Dir(alpha).
    """

    # Compute cumulative tail sums for Beta parameters
    b = np.cumsum(alpha[::-1])[::-1][1:]  # [sum_{i=j+1..K} alpha_i]
    a = alpha[:-1]

    # Inverse CDF (vectorized)
    v = beta.ppf(u, a, b)  # shape (K-1,)

    # Compute cumulative product of (1 - v)
    one_minus_v = 1.0 - v
    cumprod_one_minus_v = np.cumprod(one_minus_v)

    # Compute x components
    x = np.empty(N_pars)
    x[:-1] = v * np.concatenate(([1.0], cumprod_one_minus_v[:-1]))
    x[-1] = cumprod_one_minus_v[-1]

    return x

def get_line_angle(x, y):
    """
    Calculates the slope and angle of the best-fit line using 
    optimized vector algebra (dot products).
    
    Returns:
        slope (float)
        angle_degrees (float)
    """
    # 1. Center the data (Remove the mean)
    # This centers the cloud of points around (0,0)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    
    # 2. Calculate Slope (m)
    # Formula: m = sum(x_centered * y_centered) / sum(x_centered^2)
    # np.dot is significantly faster than .sum() for these operations
    numerator = np.dot(x_centered, y_centered)
    denominator = np.dot(x_centered, x_centered)
    
    m = numerator / denominator
    
    # 3. Convert to Degrees
    # arctan returns radians, so we convert to degrees
    angle = np.degrees(np.arctan(m))
    
    return angle

# ----------------------------------------------------------------------
# IO & Filesystem Utilities
# ----------------------------------------------------------------------

def save_path(path: str):
    """Create directory if it doesn’t exist."""
    os.makedirs(path, exist_ok=True)

def get_combinations(arr):
    """Return all pairwise combinations from a 1D array."""
    return np.array(list(combinations(arr, 2)))

# ----------------------------------------------------------------------
# Dataset Reading
# ----------------------------------------------------------------------

def read_dataset(dataset_fold: str, f1: int, f2: int, f3: int):
    """
    Read data, ranks, and pairwise comparisons from your specific dataset folder structure.
    """
    # Calculate expected number of constraints (matches generation logic)
    num_dm_dec = int(np.round(f3 * (f1 * (f1 - 1) / 200)))
    
    table_path = os.path.join(dataset_fold, f"f1_{f1}__f2_{f2}__ndm_{num_dm_dec}table.csv")
    rank_path = os.path.join(dataset_fold, f"f1_{f1}__f2_{f2}__ndm_{num_dm_dec}rank+preferences.csv")
    Us_path = os.path.join(dataset_fold, f"f1_{f1}__f2_{f2}__ndm_{num_dm_dec}Us.csv")

    if not os.path.exists(table_path) or not os.path.exists(rank_path):
        raise FileNotFoundError(f"Missing dataset files for f1={f1}, f2={f2}, f3={f3}")

    # --- Load Table ---
    # Shape: (hm * f1, f2) flattened in file, reshape to (hm, f1, f2)
    data = np.loadtxt(table_path)
    # Calculate hm (number of human models/runs) dynamically
    hm = data.shape[0] // f1
    data = data.reshape((hm, f1, f2))

    # --- Load Ranks and Preferences ---
    # Shape: (hm, f1 + 2*num_dm_dec)
    raw = np.loadtxt(rank_path)
    
    # Safety check if only 1 run exists (loadtxt returns 1D array)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    
    # 1. Ranks are the first f1 columns
    rks = raw[:, :f1]
    
    # 2. Preferences are the remaining columns
    cps_matrix = raw[:, f1:]
    
    # 3. Convert flattened preferences matrix into list of pairs
    coupless = []
    for row in cps_matrix:
        # row contains [w1, l1, w2, l2, ...]
        dm_pairs = []
        for i in range(num_dm_dec):
            # Extract consecutive pairs
            pair = row[2 * i : 2 * i + 2]
            dm_pairs.append(pair)
        coupless.append(dm_pairs)

    # --- Load True Utilities if available ---
    Us = np.loadtxt(Us_path) if os.path.exists(Us_path) else None

    return data, rks, coupless, Us

def load_test_results(test_name: str, dataset_fold: str, sub_fold: str, num_dm_dec: int, f1: int, f2: int, f3: int):
    """
    Generic function to load test results (ASRS, ASPS, AIOS).
    Handles differences in file naming conventions between metrics.
    """
    test_fold = os.path.join('tests_'+dataset_fold, sub_fold, test_name)
    sub_test_fold = os.path.join(test_fold, f"f1_{f1}_f2_{f2}_f3_{f3}")
    
    loaded_data = []
    loaded_data_active = []
    
    for i in range(1, num_dm_dec + 1):
        path = os.path.join(sub_test_fold, f"{test_name}_{i}.npy")
        path_active = os.path.join(sub_test_fold, f"{test_name}_{i}_active.npy") # Default: ASPS style
        
        # AIOS and ASRS save files as 'metric_active_i.npy'
        if test_name == 'aios' or test_name == 'asrs': 
            path_active = os.path.join(sub_test_fold, f"{test_name}_active_{i}.npy")

        if not os.path.exists(path) or not os.path.exists(path_active):
            print(path, "or", path_active, "not found. Skipping.")
            continue

        d = np.load(path)
        da = np.load(path_active)
        loaded_data.append(d.flatten())
        loaded_data_active.append(da.flatten())
        
    if not loaded_data:
        return np.array([]), np.array([])
        
    return np.stack(loaded_data, axis=0), np.stack(loaded_data_active, axis=0)

def parse_subfold_string(s):
    parts = s.split('_')
    alg_str = f"{parts[0]}-{parts[1]}" 
    active_str = parts[2]
    return alg_str, active_str