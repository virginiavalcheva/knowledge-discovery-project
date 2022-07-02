import io
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from file_reader import decodeFile
from constants import LABELS_FILENAME

def mapIdToGender(mapped_ids_to_trait):
    for key, value in mapped_ids_to_trait.items():
        if value == 'female':
            mapped_ids_to_trait[key] = 0
        if value == 'male':
            mapped_ids_to_trait[key] = 1
        if value == 'nonbinary':
            mapped_ids_to_trait[key] = 2

def mapIdToBirtyearInterval(mapped_ids_to_trait):
    for key, value in mapped_ids_to_trait.items():
        if 2000 < value <= 2012:
            mapped_ids_to_trait[key] = 0
        if 1988 < value <= 2000:
            mapped_ids_to_trait[key] = 1
        if 1976 < value <= 1988:
            mapped_ids_to_trait[key] = 2
        if 1964 < value <= 1976:
            mapped_ids_to_trait[key] = 3
        if 1952 < value <= 1964:
            mapped_ids_to_trait[key] = 4
        if 1940 <= value <= 1952:
            mapped_ids_to_trait[key] = 5

def mapIdToOccupation(mapped_ids_to_trait):
    for key, value in mapped_ids_to_trait.items():
        if value == 'sports':
            mapped_ids_to_trait[key] = 0
        if value == 'performer':
            mapped_ids_to_trait[key] = 1
        if value == 'creator':
            mapped_ids_to_trait[key] = 2
        if value == 'politics':
            mapped_ids_to_trait[key] = 3
        if value == 'manager':
            mapped_ids_to_trait[key] = 4
        if value == 'science':
            mapped_ids_to_trait[key] = 5
        if value == 'professional':
            mapped_ids_to_trait[key] = 6
        if value == 'religious':
            mapped_ids_to_trait[key] = 7

def label(trait):
    print("Labeling data based on given trait")
    mapped_ids_to_trait = {}
    resource_as_file = io.open(LABELS_FILENAME, mode="r", encoding="utf8")

    for line in decodeFile(resource_as_file):
        mapped_ids_to_trait[line['id']] = line[trait]

    if trait == 'gender':
        mapIdToGender(mapped_ids_to_trait)
    elif trait == 'birthyear':
        mapIdToBirtyearInterval(mapped_ids_to_trait)
    elif trait == 'occupation':
        mapIdToOccupation(mapped_ids_to_trait)

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
   
def birthyear_pre_processing(df):
    df[1] = np.where(df[1].between(2000,2012), 0, df[1])
    df[1] = np.where(df[1].between(1988,2000), 1, df[1])
    df[1] = np.where(df[1].between(1976,1988), 2, df[1])
    df[1] = np.where(df[1].between(1964,1976), 3, df[1])
    df[1] = np.where(df[1].between(1952,1964), 4, df[1])
    df[1] = np.where(df[1].between(1940,1952), 5, df[1])
    
def gender_pre_processing(df):
    df[2] = df[2].replace('female', 0)
    df[2] = df[2].replace('male', 1)
    df[2] = df[2].replace('nonbinary', 2)

def occupation_pre_processing(df):  
    df[3] = df[3].replace('creator', 0)
    df[3] = df[3].replace('performer', 1)
    df[3] = df[3].replace('politics', 2)
    df[3] = df[3].replace('sports', 3)
    df[3] = df[3].replace('manager', 4)
    df[3] = df[3].replace('science', 5)
    df[3] = df[3].replace('professional', 6)
    df[3] = df[3].replace('religious', 7)

