import io, json
from constants import FEEDS_FILENAME, LABELS_FILENAME, TRAIN_UPPER_LIMIT_DATA, FEEDS_100_FILENAME

def decodeFile(file):
    buffer = ""
    dec = json.JSONDecoder()
    for line in file:
        buffer = buffer.strip(" \n\r\t") + line.strip(" \n\r\t")
        while(True):
            try:
                r = dec.raw_decode(buffer)
            except:
                break
            yield r[0]
            buffer = buffer[r[1]:].strip(" \n\r\t")

def read_resource_from_file(train_user_ids, train_user_tweets):
    resource_as_file = io.open(FEEDS_FILENAME, mode="r", encoding="utf8")
    count = 0

    for line in decodeFile(resource_as_file):
        train_user_ids.append(line['id'])

        user_tweets = ""
        for tweet in line['text']:
            if tweet.isascii():
                user_tweets = user_tweets + tweet
        train_user_tweets.append(user_tweets)

        count += 1
        if count == TRAIN_UPPER_LIMIT_DATA:
            break
    resource_as_file.close()


def main():
    train_user_ids = []
    train_user_tweets = []
    read_resource_from_file(train_user_ids, train_user_tweets)
    print(train_user_ids)
    print(train_user_tweets)

if __name__ == "__main__":
    main()