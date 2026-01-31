def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    out = []
    a = float(alpha)
    for z in x:
        z = float(z)
        out.append(z if z >= 0 else a * (math.exp(z) - 1) )
    return out