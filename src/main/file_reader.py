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

def read_resource_from_file(filename, user_ids_data, user_tweets_data, lower_limit, upper_limit):
    resource_as_file = io.open(filename, mode="r", encoding="utf8")
    count = lower_limit

    for line in decodeFile(resource_as_file):
        if lower_limit <= count < upper_limit:
            user_ids_data.append(line['id'])

            user_tweets = ""
            for tweet in line['text']:
                if tweet.isascii():
                    user_tweets = user_tweets + tweet
            user_tweets_data.append(user_tweets)
            count += 1
        if count == upper_limit:
            break

    resource_as_file.close()