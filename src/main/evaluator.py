import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

def mapIdToGender(data):
    trait_ids = []
    for i in data:
        if i == 'female':
            trait_ids.append(0)
        if i == 'male':
            trait_ids.append(1)
        if i == 'nonbinary':
            trait_ids.append(2)
    return trait_ids

def mapIdToOccupation(data):
    trait_ids = []
    for i in data:
        if i == 'sports':
            trait_ids.append(0)
        if i == 'performer':
            trait_ids.append(1)
        if i == 'creator':
            trait_ids.append(2)
        if i == 'politics':
            trait_ids.append(3)
        if i == 'manager':
            trait_ids.append(4)
        if i == 'science':
            trait_ids.append(5)
        if i == 'professional':
            trait_ids.append(6)
        if i == 'religious':
            trait_ids.append(7)
    return trait_ids

def mapIdToBirtyear(data):
    trait_ids = []
    for i in data:
        if 2000 < i <= 2012:
            trait_ids.append(0)
        if 1988 < i <= 2000:
            trait_ids.append(1)
        if 1976 < i <= 1988:
            trait_ids.append(2)
        if 1964 < i <= 1976:
            trait_ids.append(3)
        if 1952 < i <= 1964:
            trait_ids.append(4)
        if 1940 <= i <= 1952:
            trait_ids.append(5)
    return trait_ids

def showPrecisionRecallCurve(test_data_ids, predicted_data_ids):
    precision, recall, _ = precision_recall_curve(test_data_ids, predicted_data_ids)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.show()

def printMetrics(test_data_ids, predicted_data_ids):
    print("Precision Score: ", precision_score(test_data_ids, predicted_data_ids, average="macro"))
    print("Recall Score: ", recall_score(test_data_ids, predicted_data_ids, average="macro"))
    score = f1_score(test_data_ids, predicted_data_ids, average="macro")
    print("F1 Score: ", score)
    return score

def evaluateGenderPredictions(test_data, predicted_data):
    test_data_ids = mapIdToGender(test_data)
    predicted_data_ids = mapIdToGender(predicted_data)

    return printMetrics(test_data_ids, predicted_data_ids)

def evaluateOccupationPredictions(test_data, predicted_data):
    test_data_ids = mapIdToOccupation(test_data)
    predicted_data_ids = mapIdToOccupation(predicted_data)

    return printMetrics(test_data_ids, predicted_data_ids)

def evaluateBirthyearPredictions(test_data, predicted_data):
    test_data_ids = mapIdToBirtyear(test_data)
    predicted_data_ids = mapIdToBirtyear(predicted_data)

    return printMetrics(test_data_ids, predicted_data_ids)