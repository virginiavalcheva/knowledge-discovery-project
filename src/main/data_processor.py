import io
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from file_reader import decodeFile
from constants import LABELS_FILENAME

def label(trait):
    print("Labeling data based on given trait")
    mapped_ids_to_trait = {}
    resource_as_file = io.open(LABELS_FILENAME, mode="r", encoding="utf8")

    for line in decodeFile(resource_as_file):
        mapped_ids_to_trait[line['id']] = line[trait]
    return mapped_ids_to_trait

def vectorize(train_user_tweets, test_user_tweets):
    print("Vectorizing data")
    vectorizer = CountVectorizer(stop_words="english")

    train_counts = vectorizer.fit_transform(train_user_tweets)

    train_tf_idf_transformer = TfidfTransformer(use_idf=False).fit(train_counts)
    train_data_tfidf = train_tf_idf_transformer.fit_transform(train_counts)

    test_counts = vectorizer.transform(test_user_tweets)

    test_tf_idf_transformer = TfidfTransformer(use_idf=False).fit(test_counts)
    test_data_tfidf = test_tf_idf_transformer.fit_transform(test_counts)

    return train_data_tfidf, test_data_tfidf