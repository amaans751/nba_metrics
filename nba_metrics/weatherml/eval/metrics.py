from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error
import math

def bin_metrics(y_true, y_prob):
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }

def reg_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse}
