from constants import FEEDS_FILENAME, TRAIN_UPPER_LIMIT_DATA, TEST_UPPER_LIMIT_DATA
from file_reader import read_resource_from_file
import naive_bayes as nb, support_vector_machine as svm, decision_tree as dt
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

    # print("\nStarting NB classifier - occupation")
    # nb.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.OCCUPATION.value)
    #
    # print("\nStarting NB classifier - gender")
    # nb.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.GENDER.value)
    #
    # print("\nStarting SVM classifier - occupation")
    # svm.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.OCCUPATION.value)
    #
    # print("\nStarting SVM classifier - gender")
    # svm.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.GENDER.value)
    #
    print("\nStarting Decision Tree classifier - birthyear")
    dt.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.BIRTHYEAR.value)

if __name__ == "__main__":
    main()
