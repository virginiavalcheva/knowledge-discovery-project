import torch
import pandas as pd
import tensorflow as tf
import io
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils.data_utils import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
from celebrity_creator import get_pre_processed_celebrities
from sklearn.metrics import f1_score
from constants import MAX_LEN, BATCH_SIZE, EPOCH_SIZE, OCCUPATION_SIZE
from evaluator import evaluateGenderPredictions

df = get_pre_processed_celebrities()

#get tweets and special tokens at the beginning and end of each sentence
def get_tweets(df):
    tweets = df[4].values
    tweets = ["[CLS] " + str(sentence) + " [SEP]" for sentence in tweets]
    return tweets

tweets = get_tweets(df)  

#get occupation
def get_occupation(df):
    occupations=df[3].values
    occupations=[int(occupation) for occupation in occupations]
    return occupations

labels = get_occupation(df)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(tweet) for tweet in tweets]
#set the text size due to resoled in an error: Token indices sequence length is longer than the specified maximum  sequence length for this BERT model (71969 > 512)
tokenized_texts=[text[:512] for text in tokenized_texts]

input_ids= [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
def create_attention_masks():
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks

attention_masks = create_attention_masks()

#use train_test_split to split our data into train and validation sets for training
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, 
                                                            random_state=2022, test_size=0.03)
  
train_masks, test_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2022, test_size=0.03)

train_masks, validation_masks, _, _ = train_test_split(train_masks, train_inputs,
                                             random_state=2022, test_size=0.03)

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(train_inputs, train_labels, 
                                                            random_state=2022, test_size=0.03)                                                       
                                                       
#convert the data into torch tensors(the required datatype)
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
test_inputs = torch.tensor(test_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
test_labels = torch.tensor(test_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)
test_masks = torch.tensor(test_masks)

#create an iterator for data
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

testing_data = TensorDataset(test_inputs, test_masks, test_labels)
testing_sampler = SequentialSampler(testing_data)
testing_dataloader = DataLoader(testing_data, sampler=testing_sampler, batch_size=BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=OCCUPATION_SIZE)
#model.cuda()
#print(torch.cuda.is_available())

device = torch.device("cpu")

def create_optimizer():
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]
    warmup_proportion = 0.1
    lr = 1e-3
    optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=warmup_proportion)
    return optimizer
  
train_loss_set = []
def tracking_variables(nb_tr_examples, nb_tr_steps):
    nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
    optimizer = create_optimizer()
    
    for step, batch in (enumerate(train_dataloader)) :
        batch = tuple(t.to(device) for t in batch)
        batch_input_ids, batch_input_mask, batch_labels = batch
        optimizer.zero_grad()
        loss = model(batch_input_ids, attention_mask=batch_input_mask, labels=batch_labels)
        train_loss_set.append(loss.item())
        loss.backward()
        optimizer.step()
    
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += batch_input_ids.size(0)
        nb_tr_steps += 1
    #print("Train loss: {}".format(tr_loss/nb_tr_steps))
    return nb_tr_examples, nb_tr_steps
    
def evaluate_data_for_epoch():
    current_result = 0.0
    step = 0
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        batch_input_ids, batch_input_mask, batch_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
          logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy().flatten()
        predicted_data = np.argmax(logits, axis=1).flatten()
        evaluateOccupationPredictions(label_ids, predicted_data)

def BERT_occupation():
    for epoch in trange(EPOCH_SIZE, desc="Epoch"):
        model.train()
        nb_tr_examples, nb_tr_steps = 0,0
        #tracking_variables(nb_tr_examples, nb_tr_steps)
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        evaluate_data_for_epoch()
