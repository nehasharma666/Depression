import pandas as pd
import numpy as np
import seaborn as sns
import warnings
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import csv
import urllib.request
import torch

sns.set(font_scale=1)
warnings.filterwarnings('ignore')
torch.cuda.device_count()

"""
Choose the ids from user_ids file and export data from SMHD paper. then select the rows with word count <3 to >200 only. 
"""
# read the data
data = pd.read_csv("depression.csv") # this data file is the selected 1316 users from depression group and 1316 users from control group which has word count greater than 3 and less then 200.
print("shape of df:",data.shape)

# Roberta sentiment model and tokenizer
sentiment_model_path = f"cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_path)

# Download label mapping
sentiment_labels = []
sentiment_mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
with urllib.request.urlopen(sentiment_mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
sentiment_labels = [row[1] for row in csvreader if len(row) > 1]


sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_path)

#sentiment_model.save_pretrained(sentiment_model_path)
#sentiment_tokenizer.save_pretrained(sentiment_model_path)

device = torch.device('cuda')
sentiment_model.to(device)

#Customer dataloader
class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data['id'].iloc[idx], self.data['text'].iloc[idx], self.data['clean_text'].iloc[idx], self.data['label'].iloc[idx], self.data['created_utc'].iloc[idx]

# BATCH SIZE CALCULATION
def find_optimal_batch_size(data_size, max_batch_size=512):
    """Find an optimal batch size such that data_size/batch_size is a whole number."""
    
    for batch_size in range(max_batch_size, 0, -1):
        if data_size % batch_size == 0:
            return batch_size
    return 1  # Return 1 if no optimal batch size is found up to max_batch_size
dataset = TextDataset(data)
data_size = len(data)
optimal_batch_size = find_optimal_batch_size(data_size)

optimal_batch_size
train_dataloader = DataLoader(dataset, batch_size=optimal_batch_size, shuffle=False,drop_last=False) # Shuffle should be set to False

# 1. Compute the total number of rows in the original DataFrame
original_data_count = len(data)

# 2. Iterate through the DataLoader and count the total number of items processed
dataloader_count = sum(len(batch[0]) for batch in train_dataloader)  # Assuming batch[0] contains the 'id' or 'text'

# 3. Compare the two counts
is_all_data_processed = original_data_count == dataloader_count

print(is_all_data_processed)


# Ensure the model is in evaluation mode
sentiment_model.eval()

# List to store the results as dictionaries
results_list = []
# Process each batch from the dataloader
for ids, doc,clean_texts, labels,utc in train_dataloader:  
    
    # Convert tensors to appropriate data structures
    doc_list =list(doc)
    utc_list = list(utc)
    labels_list = list(labels)
    texts_list = list(clean_texts)
   
    
    # Tokenize and move data to GPU
    encoded_input = sentiment_tokenizer(texts_list, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    
    # Calculate sentiment
    with torch.no_grad():
        output = sentiment_model(**encoded_input)
    
    # Compute the softmax to get the confidence scores
    scores = torch.nn.functional.softmax(output.logits, dim=1)
    
    # Extract confidence scores for each sentiment label and determine the max label
    for idm, doc,clean_text_,label_,created_utc_, score in zip(ids, doc_list,texts_list,labels_list,utc_list, scores):
           
        # Determine the sentiment label with the highest confidence
        max_idx = torch.argmax(score).item()
        max_label = sentiment_labels[max_idx]
        
        # Create a dictionary for this row and append to the results list
        row_dict = {
            'id': idm.item(),
            'label':label_,
            'document':doc,
            'created_utc':created_utc_.item(),
            'clean_text': clean_text_,
            'negative': score[0].item(),
            'neutral': score[1].item(),
            'positive': score[2].item(),
            'sentiment': max_label
        }
        results_list.append(row_dict)


# Convert the results list to a DataFrame
results_df = pd.DataFrame(results_list)
# Display the new DataFrame with predictions
results_df.head()
results_df.shape,data.shape
data.label.value_counts(), data.groupby('label')['id'].nunique()
results_df.label.value_counts(), results_df.groupby('label')['id'].nunique()
results_df.to_csv("depression_sentiments_roberta.csv",index=False) # this is the next data file to be used for topic modeling containing original data +sentiment labels
