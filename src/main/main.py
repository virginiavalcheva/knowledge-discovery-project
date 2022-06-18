from constants import FEEDS_FILENAME, TRAIN_UPPER_LIMIT_DATA, TEST_UPPER_LIMIT_DATA
from file_reader import read_resource_from_file
import naive_bayes as nb, support_vector_machine as svm


def main():
    train_user_ids = []
    train_user_tweets = []
    print("Reading train data")
    read_resource_from_file(FEEDS_FILENAME, train_user_ids, train_user_tweets, 0, TRAIN_UPPER_LIMIT_DATA)

    test_user_ids = []
    test_user_tweets = []
    print("Reading test data")
    read_resource_from_file(FEEDS_FILENAME, test_user_ids, test_user_tweets, TRAIN_UPPER_LIMIT_DATA, TEST_UPPER_LIMIT_DATA)

    # traits: birthyear, gender, occupation, fame
    print("Starting NB classifier")
    nb.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, 'occupation')

    print("Starting SVM classifier")
    svm.classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, 'gender')

if __name__ == "__main__":
    main()
