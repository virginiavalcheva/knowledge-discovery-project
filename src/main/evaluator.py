import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

def showPrecisionRecallCurve(test_data_ids, predicted_data_ids):
    precision, recall, _ = precision_recall_curve(test_data_ids, predicted_data_ids)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()

def printMetrics(test_data_ids, predicted_data_ids):
    print("Accuracy: ", accuracy_score(test_data_ids, predicted_data_ids))
    print("Precision Score Micro: ", precision_score(test_data_ids, predicted_data_ids, average="micro"))
    print("Recall Score Micro: ", recall_score(test_data_ids, predicted_data_ids, average="micro"))
    score = f1_score(test_data_ids, predicted_data_ids, average="micro")
    print("F1 Score Micro: ", score)

    print("Precision Score Macro: ", precision_score(test_data_ids, predicted_data_ids, average="macro"))
    print("Recall Score Macro: ", recall_score(test_data_ids, predicted_data_ids, average="macro"))
    score = f1_score(test_data_ids, predicted_data_ids, average="macro")
    print("F1 Score Macro: ", score)

    return score