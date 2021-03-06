from constants import FEEDS_FILENAME, TRAIN_UPPER_LIMIT_DATA, TEST_UPPER_LIMIT_DATA
from file_reader import read_resource_from_file
import multinomial_naive_bayes as nb, support_vector_machine as svm, \
    decision_tree as dt, linear_regression as lr, random_forest as rf, \
    logistic_regression as log_r, bernoulli_naive_bayes as ber_nb
import evaluator as ev
from enum import Enum
import naive_bayes as nb, support_vector_machine as svm
from celebrity import getCelebrities
from bert_occupation import BERT_occupation()
from bert_gender import BERT_gender()
from bert_birthyear import BERT_birthyear()

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
    print("Starting Multinominal NB classifier - OCCUPATION")
    test_data, predicted_data = nb.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.OCCUPATION.value)
    f1_occupation = ev.printMetrics(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("Starting SVM classifier - GENDER")
    test_data, predicted_data = svm.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.GENDER.value)
    f1_gender = ev.printMetrics(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("Starting Decision Tree classifier - GENDER")
    test_data, predicted_data = dt.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.GENDER.value)
    f1_gender = ev.printMetrics(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("Starting Logistic Regression classifier - GENDER")
    test_data, predicted_data = log_r.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.GENDER.value)
    f1_gender = ev.printMetrics(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("Starting Bernoulli NB classifier - GENDER")
    test_data, predicted_data = ber_nb.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.GENDER.value)
    f1_gender = ev.printMetrics(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("Starting Linear Regression classifier - BIRTHYEAR")
    test_data, predicted_data = lr.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.BIRTHYEAR.value)
    f1_birthyear = ev.printMetrics(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("Starting Decision Tree classifier - BIRTHYEAR")
    test_data, predicted_data = dt.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.BIRTHYEAR.value)
    f1_birthyear = ev.printMetrics(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    print("Starting Random Forest classifier - BIRTHYEAR")
    test_data, predicted_data = rf.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, Trait.BIRTHYEAR.value)
    f1_birthyear = ev.printMetrics(test_data, predicted_data)
    print("---------------------------------------------------------------------------\n")

    cRank = 3 / ((1 / f1_occupation) + (1 / f1_gender) + (1 / f1_birthyear))
    print("cRank: ", cRank)

    print("Starting BERT model - BIRTHYEAR")
    BERT_birthyear()
    print("---------------------------------------------------------------------------\n")
    
    print("Starting BERT model - GENDER")
    BERT_gender()
    print("---------------------------------------------------------------------------\n")
    
    print("Starting BERT model - OCCUPATION")
    BERT_occupation()
    print("---------------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
