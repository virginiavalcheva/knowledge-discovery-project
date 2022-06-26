import io
import pandas as pd 
import numpy as np
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

