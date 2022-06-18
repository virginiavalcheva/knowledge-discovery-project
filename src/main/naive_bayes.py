import io, numpy as np
from sklearn.naive_bayes import MultinomialNB
from data_processor import vectorize, label

def classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, trait):
    #traits: birthyear, gender, occupation, fame
    mapped_ids_to_trait = label(trait)

    trait_train = []
    for id in train_user_ids:
        trait_train.append(mapped_ids_to_trait.get(id))

    train_data_tfidf = vectorize(train_user_tweets)

    print("Starting classifier")
    classifier = MultinomialNB().fit(train_data_tfidf, trait_train)

    trait_test = []
    for id in test_user_ids:
        trait_test.append(mapped_ids_to_trait.get(id))

    test_data_tfidf = vectorize(test_user_tweets)

    print("Predicting")
    predicted_trait = classifier.predict(test_data_tfidf)
    print("NBC: Occupation: ", np.mean(predicted_trait == trait_test))
