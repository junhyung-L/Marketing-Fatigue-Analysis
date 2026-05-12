import numpy as np

def compute_roc_auc_manual(y_true, y_prob):
    """
    Manually calculates ROC-AUC using trapezoidal integration.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    mask = ~np.isnan(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]

    cum_pos = np.cumsum(y_true_sorted == 1)
    cum_neg = np.cumsum(y_true_sorted == 0)

    tpr = cum_pos / n_pos
    fpr = cum_neg / n_neg

    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0
    for i in range(len(y_true_sorted)):
        auc += (fpr[i] - prev_fpr) * (tpr[i] + prev_tpr) / 2.0
        prev_fpr = fpr[i]
        prev_tpr = tpr[i]

    return float(auc)

def compute_aic_bic_manual(y_true, y_prob, n_features):
    """
    Calculates AIC and BIC based on Log-Likelihood.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    
    log_likelihood = np.sum(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    
    n = len(y_true)
    k = n_features
    
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    return float(aic), float(bic), float(log_likelihood)

def compute_classification_metrics(y_true, y_prob, thr=0.5, n_features=None):
    """
    Calculates accuracy, precision, recall, specificity, F1, and optionally AIC/BIC.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= thr).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    n = len(y_true)
    eps = 1e-9

    accuracy    = (tp + tn) / (n + eps)
    precision   = tp / (tp + fp + eps)
    recall      = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    f1          = 2 * precision * recall / (precision + recall + eps)

    roc_auc = compute_roc_auc_manual(y_true, y_prob)

    metrics = {
        "accuracy":       float(accuracy),
        "precision":      float(precision),
        "recall":         float(recall),
        "specificity":    float(specificity),
        "f1":             float(f1),
        "roc_auc":        float(roc_auc),
        "n_eval":         int(n),
        "pos_rate_eval":  float(y_true.mean()),
    }

    if n_features is not None:
        aic, bic, ll = compute_aic_bic_manual(y_true, y_prob, n_features)
        metrics["aic"] = aic
        metrics["bic"] = bic
        metrics["log_likelihood"] = ll
        metrics["n_features"] = int(n_features)

    cm = (tn, fp, fn, tp)
    return metrics, cm
