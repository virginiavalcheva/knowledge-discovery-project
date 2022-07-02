from sklearn.ensemble import RandomForestClassifier
from data_processor import vectorize, label

def classify(train_user_ids, train_user_tweets, test_user_ids, test_user_tweets, trait):
    mapped_ids_to_trait = label(trait)

    trait_train = []
    for id in train_user_ids:
        trait_train.append(mapped_ids_to_trait.get(id))

    trait_test = []
    for id in test_user_ids:
        trait_test.append(mapped_ids_to_trait.get(id))

    train_data_tfidf, test_data_tfidf = vectorize(train_user_tweets, test_user_tweets)

    print("Starting Random Forest classifier")
    classifier = RandomForestClassifier(max_depth=5000, random_state=0)
    classifier.fit(train_data_tfidf, trait_train)

    print("Predicting")
    predicted_trait = classifier.predict(test_data_tfidf)

    return trait_test, predicted_trait