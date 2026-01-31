
import math
def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """
    # Write code here
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    def safe_div(num, den):
        return 0.0 if den == 0 else num / den

    st = str(system_type).strip().lower()
    metrics = {}

    if st == "classification":
        yt = [1 if int(v) == 1 else 0 for v in y_true]
        yp = [1 if int(v) == 1 else 0 for v in y_pred]
        n = len(yt)
        TP = sum(1 for a, b in zip(yt,yp) if a == 1 and b == 1)
        TN = sum(1 for a, b in zip(yt,yp) if a == 0 and b == 0)
        FP = sum(1 for a, b in zip(yt, yp) if a == 0 and b == 1)
        FN = sum(1 for a, b in zip(yt, yp) if a == 1 and b == 0)
        acc = safe_div(TP + TN, n)
        pre = safe_div(TP, TP + FP)
        recall = safe_div(TP, TP + FN)
        f1 = safe_div(2*pre*recall, pre + recall)

        metrics["accuracy"] = acc
        metrics["precision"] = pre
        metrics["recall"] = recall
        metrics["f1"] = f1

    elif st == "regression":
        yt = [float(v) for v in y_true]
        yp = [float(v) for v in y_pred]
        n = len(yt)

        if n == 0:
            mae = 0
            rmse = 0
        else:
            abs_err = [abs(a - b) for a,b in zip(yt,yp)]
            sq_err = [(a - b)**2 for a,b in zip(yt,yp)]
            mae = sum(abs_err) / n
            rmse = math.sqrt(sum(sq_err) / n)

        metrics["mae"] = mae
        metrics["rmse"] = rmse
    
    elif st == "ranking":
        yt = [1 if int(v) == 1 else 0 for v in y_true]
        yp = [float(v) for v in y_pred]

        n = len(yt)
        k = 3
        if n == 0:
            rel_in_topk = 0
        else:
            order = sorted(range(n), key=lambda i: yp[i], reverse=True)
            topk = order[:k]
            rel_in_topk = sum(yt[i] for i in topk)

        total_relevant = sum(yt)
        metrics["precision_at_3"] = safe_div(rel_in_topk, k)
        metrics["recall_at_3"] = safe_div(rel_in_topk, total_relevant)

    else:
        raise ValueError("Unknown system_type")

    return sorted(metrics.items(), key=lambda kv: kv[0])
