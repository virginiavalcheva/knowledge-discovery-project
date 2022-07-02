import pandas as pd 
import io
from file_reader import decodeFile, read_resource_from_file
from data_processor import birthyear_pre_processing, occupation_pre_processing, gender_pre_processing
from constants import FEEDS_FILENAME, TRAIN_UPPER_LIMIT_DATA, LABELS_FILENAME

ids, tweet = read_resource_from_file(FEEDS_FILENAME, 0, TRAIN_UPPER_LIMIT_DATA)

# (id->0, birthyear->1, gender->2, occupation->3, text->4)
def create_celebrities():
    entries=[]
    for id_ in range(len(ids)):
        resource_as_file = io.open(LABELS_FILENAME, mode="r", encoding="utf8")
        for line in decodeFile(resource_as_file):
            if(line['id'] == id_):
                entry=(line['id'], line['birthyear'], line['gender'], line['occupation'], tweet[id_])
                entries.append(entry)
    df = pd.DataFrame(entries)
    #df.to_csv("celebrity_profiling_training-dataset.csv") 
    #print(df.shape)
    return df

def get_celebrities():
    return create_celebrities()

def get_pre_processed_celebrities():
    df = create_celebrities()
    #data pre-processing
    birthyear_pre_processing(df)
    gender_pre_processing(df)
    occupation_pre_processing(df)
    return df

