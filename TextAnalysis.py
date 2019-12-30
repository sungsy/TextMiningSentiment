# %%
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

# %%
tmp_pos_label = np.full((2500,), 1)
tmp_neg_label = np.full((2500,), 0)
# %%
positive_text = pd.read_csv('data/pos.txt', header=None, names=['text'])
positive_label = pd.DataFrame(data=tmp_pos_label, columns=['label'])
negative_text = pd.read_csv('data/neg.txt', header=None, names=['text'])
negative_label = pd.DataFrame(data=tmp_neg_label, columns=['label'])
# %%
pos_txt_tknzd = positive_text.text.apply(word_tokenize)
# %%
pos_txt_list = []
for x in pos_txt_tknzd:
    pos_txt_list.extend(x)

# %%
fdist = FreqDist(pos_txt_list)
# %%
fdist.plot(60, cumulative=False)
plt.show()
# %%
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer

# tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
text_counts_positive = cv.fit_transform(positive_text['text'])
text_counts_negative = cv.fit_transform(negative_text['text'])
