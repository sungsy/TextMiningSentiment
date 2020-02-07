# %%
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

import IOHandler as io

# %%
# Reading data from csv & combining into labeled dataframes

positive_text_data = io.read_data(src_data='data/pos.txt')
negative_text_data = io.read_data(src_data='data/neg.txt')

# One big dataframe containing all positive (=1) and negative (=0) text phrases
text_data = pd.concat((positive_text_data, negative_text_data), axis=0)

# %%
# loading data
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
# Model Generation Using Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

clf_MNB = MultinomialNB().fit(X_train, y_train)
predicted_MNB = clf_MNB.predict(X_test)
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted_MNB))

# %%
# Train RandomForest Classificator
from sklearn.ensemble import RandomForestClassifier
clf_RF = RandomForestClassifier().fit(X_train, y_train)
predicted_RF = clf_RF.predict(X_test)
print("Random Forest Accuracy:", metrics.accuracy_score(y_test, predicted_RF))

# %%
# Train Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
clf_LogReg = LogisticRegression().fit(X_train, y_train)
predicted_LogReg = clf_LogReg.predict(X_test)
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted_LogReg))

# %%
# Train Linear SVC Classifier
from sklearn.svm import LinearSVC
clf_SVM = LinearSVC().fit(X_train, y_train)
predicted_SVM = clf_SVM.predict(X_test)
print("Linear SVC Classifier:", metrics.accuracy_score(y_test, predicted_SVM))

# %%
# https://streamhacker.com/2012/11/22/text-classification-sentiment-analysis-nltk-scikitlearn/
print("Random Forest Accuracy:", metrics.accuracy_score(y_test, predicted_RF))
print("MultinomialNB Accuracy:", metrics.accuracy_score(y_test, predicted_MNB))
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, predicted_LogReg))
print("Linear SVC Classifier:", metrics.accuracy_score(y_test, predicted_SVM))

2# %%
# Read Evaluation Data from TXT file
evaluation_data = pd.read_csv('data/evaluation.txt', names=['text'])

# %%
# Apply Tokenizer, Stemming and Bag of Words/ One Hot Encoding Maker
text_counts_eval = cv.transform(evaluation_data['text'])
# %%
# Predict Evaluation Data
eval_predict = clf_LogReg.predict(text_counts_eval)
eval_predict = pd.DataFrame(data=eval_predict, columns=['label'])
eval_predict_results = pd.concat((evaluation_data, eval_predict), axis=1)

# %%
# Write Results into csv
io.write_predicted_data(eval_predict)


