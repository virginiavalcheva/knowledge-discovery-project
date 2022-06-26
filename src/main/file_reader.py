import io, json

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

def read_resource_from_file(filename, lower_limit, upper_limit):
    user_ids_data = []
    user_tweets_data = []
    resource_as_file = io.open(filename, mode="r", encoding="utf8")
    count = 0

    for line in decodeFile(resource_as_file):
        if lower_limit <= count < upper_limit:
            user_ids_data.append(line['id'])

            user_tweets = ""
            for tweet in line['text']:
                if tweet.isascii():
                    user_tweets = user_tweets + tweet
            user_tweets_data.append(user_tweets)
        if count == upper_limit:
            break
        count += 1

    resource_as_file.close()
    return user_ids_data, user_tweets_data