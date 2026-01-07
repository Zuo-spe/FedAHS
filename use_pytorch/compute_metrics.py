from sklearn.metrics import confusion_matrix
from torchmetrics.classification import BinaryAUROC
from torchmetrics.functional import accuracy, f1_score
import numpy as np

def metrics_compute(preds_sum, test_label):
    auroc = BinaryAUROC()
    acc = accuracy(preds_sum, test_label, task='binary').item()
    F1_score = f1_score(preds_sum, test_label, task='binary').item()
    AUC = auroc(preds_sum, test_label).item()
    tn, fp, fn, tp = confusion_matrix(test_label.cpu(), preds_sum.cpu()).ravel()
    TPR = tp / (tp + fn)
    TNR = tn / (tn + fp)
    G_mean = np.sqrt(TPR * TNR)  # G-mean
    return G_mean, F1_score, AUC


