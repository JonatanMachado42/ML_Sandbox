import numpy as np 


def accuracy_score(y_pred: np.ndarray, y_true: np.ndarray ) -> float :
    return (y_true == y_pred).mean() 


def confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray ) -> float :
    """
    TP -> true  positive : y_pred[N] == y_true[N] == 1
    TN -> true  negative : y_pred[N] == y_true[N] == 0
    FP -> false positive : y_pred[N] == 1 != y_true[N] == 0
    FN -> false negative : y_pred[N] == 0 != y_true[N] == 1
    """

    tp = 0 
    tn = 0 
    fp = 0
    fn = 0 

    # TODO         