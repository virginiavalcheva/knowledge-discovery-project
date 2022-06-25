import io, numpy as np
from sklearn import tree
from data_processor import vectorize, label
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, trait):
    #traits: birthyear, gender, occupation, fame
    mapped_ids_to_trait = label(trait)

    trait_train = []
    for id in train_user_ids:
        trait_train.append(mapped_ids_to_trait.get(id))

    trait_test = []
    for id in test_user_ids:
        trait_test.append(mapped_ids_to_trait.get(id))

    train_data_tfidf, test_data_tfidf = vectorize(train_user_tweets, test_user_tweets)

    print("Starting classifier")
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(train_data_tfidf, trait_train)

    print("Predicting")
    predicted_trait = classifier.predict(test_data_tfidf)
    print("Accuracy: ", accuracy_score(trait_test, predicted_trait))
    print(trait_test)
    print(predicted_trait)
    x = [i for i in predicted_trait]
    print(x)
    print("Precision: ", precision_score(trait_test, x), average="binary", pos_label="pos")
    print("Recall: ", recall_score(trait_test, x))
    print("F1 Score: ", f1_score(trait_test, predicted_trait))