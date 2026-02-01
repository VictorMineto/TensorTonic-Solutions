import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    n = int(n)
    p = float(p)
    if n < 0:
        raise ValueError("n doit etre positif")
    if not (0 <= p <= 1):
        raise ValueError("p doit etre une proba")
    
    k_arr = np.asarray(k)
    k_int = k_arr.astype(int)

    valid = (k_int >= 0) and (k_int <= n)

    PMF = np.zeros_like(k_arr, dtype=float)
    if np.any(valid):
        kv = k_int[valid]
        PMF[valid] = comb(n, kv, exact=False) * (p ** kv) * ((1 - p) ** (n - kv))


    CDF = np.zeros_like(k_arr, dtype=float)
    it = np.nditer(k_arr, flags=["multi_index"])
    while not it.finished:
        kk = int(it[0])
        if kk < 0:
            CDF[it.multi_index] = 0.0
        elif kk >= n:
            CDF[it.multi_index] = 1.0
        else :
            i = np.arange(0, kk + 1, dtype=int)
            CDF[it.multi_index] = np.sum(comb(n, i, exact=False) * (p ** i) * ((1.0 - p) ** (n - i)))
        it.iternext()

    
    if np.isscalar(k):
        return float(PMF), float(CDF)
    return PMF, CDF