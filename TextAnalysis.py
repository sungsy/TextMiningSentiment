# %%
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

# %%
# Temporary Arrays to create labels for text phrases
tmp_pos_label = np.full((2500,), 1)
tmp_neg_label = np.full((2500,), 0)
# %%
# Reading data from csv & combining into labeled dataframes
positive_text = pd.read_csv('data/pos.txt', header=None, names=['text'])
positive_label = pd.DataFrame(data=tmp_pos_label, columns=['label'])
negative_text = pd.read_csv('data/neg.txt', header=None, names=['text'])
negative_label = pd.DataFrame(data=tmp_neg_label, columns=['label'])

positive_text_label = pd.concat((positive_text, positive_label), axis=1)
negative_text_label = pd.concat((negative_text, negative_label), axis=1)

# One big dataframe containing all positive (=1) and negative (=0) text phrases
text_data = pd.concat((positive_text_label, negative_text_label), axis=0)

# %%

pos_txt_tknzd = positive_text.text.apply(word_tokenize)
pos_txt_list = []
for x in pos_txt_tknzd:
    pos_txt_list.extend(x)
# %%
fdist = FreqDist(pos_txt_list)
fdist.plot(60, cumulative=False)
plt.show()
# %%
Sentiment_count = text_data.groupby('label').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['text'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()

# %%
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# tokenizer to remove unwanted elements from our data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
text_counts = cv.fit_transform(text_data['text'])

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    text_counts, text_data['label'], test_size=0.3, random_state=1)
# %%
from sklearn.naive_bayes import MultinomialNB
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted = clf.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted))

