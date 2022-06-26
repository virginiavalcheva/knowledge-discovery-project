from constants import FEEDS_FILENAME, TRAIN_UPPER_LIMIT_DATA, TEST_UPPER_LIMIT_DATA
from file_reader import read_resource_from_file
import naive_bayes as nb, support_vector_machine as svm, decision_tree as dt
import evaluator as ev
from enum import Enum

class Trait(Enum):
    BIRTHYEAR = 'birthyear'
    GENDER = 'gender'
    OCCUPATION = 'occupation'
    FAME = 'fame'

def main():
    print("Reading train data")
    train_user_ids, train_user_tweets = read_resource_from_file(FEEDS_FILENAME, 0, TRAIN_UPPER_LIMIT_DATA)

    print("Reading test data")
    test_user_ids, test_user_tweets = read_resource_from_file(FEEDS_FILENAME, TRAIN_UPPER_LIMIT_DATA, TEST_UPPER_LIMIT_DATA)

    print("---------------------------------------------------------------------------\n")
    print("Starting NB classifier - OCCUPATION")
    test_data, predicted_data = nb.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.OCCUPATION.value)
    f1_occupation = ev.evaluateOccupationPredictions(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("---------------------------------------------------------------------------\n")
    print("Starting NB classifier - GENDER")
    test_data, predicted_data = nb.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.GENDER.value)
    f1_occupation = ev.evaluateGenderPredictions(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("\nStarting NB classifier - BIRTHYEAR")
    test_data, predicted_data = nb.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.BIRTHYEAR.value)
    f1_gender = ev.evaluateBirthyearPredictions(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("\nStarting SVM classifier - OCCUPATION")
    test_data, predicted_data = svm.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.OCCUPATION.value)
    f1_gender = ev.evaluateOccupationPredictions(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("Starting SVM classifier - GENDER")
    test_data, predicted_data = svm.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.GENDER.value)
    f1_gender = ev.evaluateGenderPredictions(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("Starting Decision Tree classifier - BIRTHYEAR")
    test_data, predicted_data = dt.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.BIRTHYEAR.value)
    f1_birthyear = ev.evaluateBirthyearPredictions(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    #cRank = 3 / ((1 / f1_occupation) + (1 / f1_gender) + (1 / f1_birthyear))
    #print("cRank: ", cRank)

if __name__ == "__main__":
    main()
