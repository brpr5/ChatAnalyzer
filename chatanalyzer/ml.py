"""
Module for machine learning
"""
import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import pickle 
import pyLDAvis


from pyLDAvis import sklearn as sklearn_lda
from sklearn.feature_extraction.text import CountVectorizer
from data import transform_dataframe, create_dataframe
from sklearn.decomposition import LatentDirichletAllocation as LDA

sns.set_style('whitegrid')
PATH = os.path.abspath(os.path.dirname(__file__))

# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

def get_topics(filename=None,number_topics=10,number_words=10, **kwargs):
    """
    Function that uses LDA to get topics swith keywords
    """
    if filename == None:
        filename = os.listdir(os.path.join(PATH,"..","raw"))[0]
    # filename = "chat_20200711.txt"
    file_path = os.path.join(PATH,"..","raw", filename)
    df = transform_dataframe(create_dataframe(file_path), scramble=False)

    # Remove media and deleted messages 
    df = df[(df['is_media'] == False) & (df['is_deleted'] == False)]

    # group by hour
    messages = df.groupby(['person','day','month','hour'])['message'].apply(lambda x: ' '.join(list(x)))

    # Only message and person columns
    df = df[['person','message']].dropna(subset =['person'])
    # Remove punctuation
    df['message'] = df['message'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert to lowercase
    df['message'] = df['message'].map(lambda x: x.lower())

    # Download stopwords
    nltk.download("stopwords")
    #Initialise the count vectorizer with the English stop words
    stopwords = nltk.corpus.stopwords.words('portuguese')
    count_vectorizer = CountVectorizer(stop_words=stopwords)
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(messages)#df['message'])

    # Visualise the 10 most common words
    # plot_10_most_common_words(count_data, count_vectorizer)

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    print_topics(lda, count_vectorizer, number_words)


    LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(number_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)

    # Create output path
    output_dir = os.path.join(PATH,"..",filename.split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    # Save, Note that the output file needs to be opened in binary mode
    with open(os.path.join(output_dir,LDAvis_data_filepath), 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

    # Load the pre-prepared pyLDAvis data from disk
    with open(os.path.join(output_dir,LDAvis_data_filepath), 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, os.path.join(output_dir,'./ldavis_prepared_'+ str(number_topics) + '.html'))



if __name__ == "__main__":
    #df = transform_dataframe(create_dataframe("raw/chat_20200711.txt"), scramble=False)
    get_topics()
    # TODO Atualizar requirements.txt!
    # todo search for best topic numbers
