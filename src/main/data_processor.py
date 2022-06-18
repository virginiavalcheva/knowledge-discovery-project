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

def vectorize(user_tweets):
    print("Vectorizing data")
    vectorizer = CountVectorizer(stop_words="english")
    counts = vectorizer.fit_transform(user_tweets)

    tf_idf_transformer = TfidfTransformer(use_idf=False).fit(counts)
    return tf_idf_transformer.fit_transform(counts)