import pandas as pd
import openai
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
from sentence_transformers import SentenceTransformer
import pickle
from umap import UMAP
from hdbscan import HDBSCAN

from sklearn.feature_extraction.text import CountVectorizer


#Read data
data = pd.read_csv("depression_sentiments_roberta.csv") # this data file is created "1. roberta_sentiment_analysis.py".
print(data.shape)
print(data.groupby('user').ids.nunique())

abstracts = data['clean_text'].astype('str')

# Pre-calculate embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = embedding_model.encode(abstracts, show_progress_bar=True)

'''with open('doc_embedding.pickle', 'wb') as pkl:
    pickle.dump(embeddings, pkl)'''

'''with open('doc_embedding.pickle', 'rb') as pkl:
    embedding = pickle.load(pkl)'''

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42,low_memory=True)

hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# KeyBERT
keybert_model = KeyBERTInspired()

# Part-of-Speech
#pos_model = PartOfSpeech("en_core_web_sm")

# MMR
mmr_model = MaximalMarginalRelevance(diversity=0.3)


# All representation models
representation_model = {
    "KeyBERT": keybert_model,
    "MMR": mmr_model,
    #"POS": pos_model
}


topic_model = BERTopic(

  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

topics,prob = topic_model.fit_transform(abstracts, embedding)
topic_model.get_topic_info()


df = data.copy()
posts = abstracts.to_list()
ids = df['ids'].to_list()
sentiments = df['sentiments'].to_list()
users = df['user'].to_list()
created_utc = df['created_utc'].to_list()
negative = df['negative'].to_list()
neutral = df['neutral'].to_list()
positive = df['positive'].to_list()


# Reduce outliers with pre-calculate embeddings instead
new_topics = topic_model.reduce_outliers(abstracts, topics, strategy="c-tf-idf")
topic_model.update_topics(abstracts, topics=new_topics)
topic_model.get_topic_info()


# reduce topics
topic_model.reduce_topics(abstracts, nr_topics='auto')

# Access updated topics
topics = topic_model.topics_
topic_model.get_topic_info()

d = pd.DataFrame({'ids':ids,'clean_text': posts,'topic': topics, 'probs':prob,'sentiments':sentiments,
                   'users':users,'created_utc':created_utc,'negative':negative,'neutral':neutral,
                 'positive':positive})

d.to_csv("Topic_df_with_sentiments.csv",index=False) # this new datafile contains original data + sentiment labels + topic modeling result

topics_df = pd.DataFrame(topic_model.get_topic_info())
topics_df.to_csv("Topic_representation.csv",index=False) # this is extra datafile from topic modeling which contains topic representation i.e. 10 words representation,which help us to explore the topics.

embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
topic_model.save("my_model_dir", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
# Load from directory
loaded_model = BERTopic.load("model_dir")
loaded_model.get_topic_info()
