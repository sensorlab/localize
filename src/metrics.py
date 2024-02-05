import numpy as np

# def mean_percentage_error(A: np.ndarray, F: np.ndarray) -> np.float64:
#     A_abs, F_abs = np.abs(A), np.abs(F)
#     mask = ~(np.maximum(A_abs, F_abs) == 0)
#     return 100.0 * np.mean((np.abs(A - F) / np.maximum(A_abs, F_abs))[mask])


def mean_percentage_error(A: np.ndarray, F: np.ndarray) -> np.float64:
    top = np.abs(A - F)
    bottom = np.maximum(np.maximum(np.abs(A), np.abs(F)), np.finfo(float).eps)
    return np.mean(top / bottom)
