# Function for calculating the evaluation metrics from the confusion matrix
# Referred from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html


def confusion_metrics(conf_matrix):
    # Accessing the elements of the confusion matrix
    # TP:True Positives, TN:True Negatives, FP:False Positives, FN:False Negatives
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    print("Elements of Confusion matrix:")
    print('True Positives:', TP)
    print('True Negatives:', TN)
    print('False Positives:', FP)
    print('False Negatives:', FN)

    # calculate accuracy
    conf_accuracy = (float(TP + TN) / float(TP + TN + FP + FN))
    # calculate mis-classification/error (Top-1 error)
    conf_misclassification = 1 - conf_accuracy
    # calculate the sensitivity
    conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
    conf_specificity = (TN / float(TN + FP))
    # calculate precision
    conf_precision = (TN / float(TN + FP))
    # calculate f_1 score
    conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
    # Printing all the calculated metric values
    print("Evaluation Metrics:")
    print(f'Accuracy(%): {(round(conf_accuracy, 2)*100)}')
    print(f'Mis-Classification/Error(%): {(round(conf_misclassification, 2)*100)}')
    print(f'Sensitivity(Recall): {round(conf_sensitivity, 2)}')
    print(f'Specificity: {round(conf_specificity, 2)}')
    print(f'Precision: {round(conf_precision, 2)}')
    print(f'f_1 Score: {round(conf_f1, 2)}')
