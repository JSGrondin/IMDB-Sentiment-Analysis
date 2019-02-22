# Importing relevant packages
import pandas as pd
import numpy as np
from data_load import get_training_data
from data_load import get_test_data
from data_load import report
from data_load import lower_text
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as randint
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from textprocessing import special_characters
from textprocessing import remove_urls
from textprocessing import remove_emails
from textprocessing import lemmatization_stem_and_stopwords
from textprocessing import char_preprocessing
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.metrics import accuracy_score

# Extracting raw features (X) and targets(y)
X_train, y_train = get_training_data('train')
X_test, X_test_Id = get_test_data('test')

# Removing URLs
X_train = remove_urls(X_train)
X_text = remove_urls(X_test)

# Removing email addresses
X_train = remove_emails(X_train)
X_test = remove_emails(X_text)

# Setting Seed
seed = 123


#Splitting them so as to have validation sets - this is only used for
# finding the optimial hyperparameters for each classifier and assessing
# different ensembling stacking metamodels. Then we retrain
# on the full training + validation set
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
#                                                  train_size=0.9, \
#                                                  test_size=0.1,
#                                                  random_state=seed)

###########################################################
#              VADER sentiment Analyzer                   #
###########################################################
# The VADER sentiment tool was developed by C.J. Huto & al.
#https://github.com/cjhutto/vaderSentiment and is a rule based tool based on
# a sentiment lexicon. Punctuation, capital letters and special characters are
# needed as they can "boost" the sentiment of certain words.

nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()
y_vader_pred_train = np.zeros(len(y_train))
# y_vader_pred_val = np.zeros(len(y_val))
y_vader_pred = np.zeros(len(X_test))


def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = vader.polarity_scores(text)
    return 1 if score['pos'] > score['neg'] else 0


for i, text in enumerate(X_train):
    y_vader_pred_train[i] = vader_polarity(text)

# for i, text in enumerate(X_val):
#     y_vader_pred_val[i] = vader_polarity(text)

# accuracy_score(y_val, y_vader_pred_val) # 0.6944

for i, text in enumerate(X_test):
    y_vader_pred[i] = vader_polarity(text)

###########################################################
#                  Text Pre-processing                    #
###########################################################

# Removing capital letters, i.e lowering all strings
X_train = lower_text(X_train)
# X_val = lower_text(X_val)
X_test = lower_text(X_test)

# Removing special characters
X_train = char_preprocessing(X_train)
# X_val = char_preprocessing(X_val)
X_test = char_preprocessing(X_test)

# with open('your_file.txt', 'w', encoding="utf-8") as f:
#     for item in X_train:
#         f.write("%s\n" % item)

# Processing text features on Xtrain, X_val and X_test
X_train = lemmatization_stem_and_stopwords(X_train,True,True,False)
# X_val = lemmatization_stem_and_stopwords(X_val,True,True,False)
X_test = lemmatization_stem_and_stopwords(X_test,True,True,False)


###########################################################
#                  Logistic Regression                    #
###########################################################

# Building Logistic Regression pipeline
pclf_LR = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,2), max_df=0.5, binary=False)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', LogisticRegression(C=2, dual=True))
])

# params = {"vect__ngram_range": [(1,1),(1,2),(1,3)],
#           "vect__max_df": [0.5, 0.75, 1],
#           "vect__binary": [False],
#           "tfidf__use_idf": [True, False],
# #          "clf__alpha": uniform(1e-2, 1e-3)
#           "clf__C": [0.8, 1, 2],
#           "clf__dual": [True, False]
#           }

# Perform randomized search CV to find best hyperparameters
# random_search = RandomizedSearchCV(pclf_LR, param_distributions = params,
#                                    cv=3,
#                                    verbose = 10, random_state = seed,
#                                    n_iter = 1)
# random_search.fit(X_train, y_train)

# Report results
# report(random_search.cv_results_)

# Assessing model performance on validation set
# y_pred = random_search.predict(X_val)
# print(metrics.classification_report(y_val, y_pred, digits=4))

# Model with rank:1
#Mean validation score: 0.886 (std: 0.000)
#Parameters: {'vect__ngram_range': (1, 2), 'vect__max_df': 0.5,
# 'vect__binary': False, 'tfidf__use_idf': True, 'clf__dual': True, 'clf__C': 2}

# Training optimal logistic regression model
pclf_LR.fit(X_train, y_train)
# y_LR_pred_train = pclf_LR.predict(X_train) # required for ensembling method
# y_LR_pred_val = pclf_LR.predict(X_val)
# accuracy_score(y_val, y_LR_pred_val)  #0.8896
y_LR_pred = pclf_LR.predict(X_test)

###########################################################
#                  Linear SVC Pipeline                    #
###########################################################

pclf_SVC = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,2), max_df=0.25, binary=False)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', LinearSVC(C=1))
])

# params = {"vect__ngram_range": [(1, 2), (1, 3)],
#           "vect__max_df": [0.25, 0.5, 0.75, 1],
#           "vect__binary": [False],
#           "tfidf__use_idf": [True],
#           "clf__C": [0.25, 0.5, 0.8, 1, 2],
#           }

# Perform randomized search CV to find best hyperparameters
# random_search = RandomizedSearchCV(pclf_SVC, param_distributions = params,
#                                    cv=2,
#                                    verbose = 10, random_state = seed,
#                                    n_iter = 25)
# random_search.fit(X_train, y_train)

# Report results
# report(random_search.cv_results_)

# Assessing model performance on validation set
# y_pred = random_search.predict(X_val)
# print(metrics.classification_report(y_val, y_pred, digits=4))

# Model with rank:1
#Mean validation score: 0.898 (std: 0.000)
#Parameters: {'vect__ngram_range': (1, 2), 'vect__max_df': 0.25,
# 'vect__binary': False, 'tfidf__use_idf': True, clf__C': 1}

# Training optimal logistic regression model
pclf_SVC.fit(X_train, y_train)
# y_LSVC_pred_train = pclf_SVC.predict(X_train) # required for ensembling method
# y_LSVC_pred_val = pclf_SVC.predict(X_val)
# accuracy_score(y_val, y_LSVC_pred_val)  #0.9032
y_LSVC_pred = pclf_SVC.predict(X_test)
my_prediction = pclf_SVC.predict(X_test)

###########################################################
#                Multinomial Naive Bayes                  #
###########################################################

pclf_NB = Pipeline([
    ('vect', CountVectorizer(ngram_range=(1,3), max_df=0.5, binary=False)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('clf', MultinomialNB(alpha=0.0107))
])

# params = {"vect__ngram_range": [(1,2),(1,3)],
#           "vect__max_df": [0.25, 0.5, 1],
#           "vect__binary": [False, True],
#           "tfidf__use_idf": [True, True],
#           "clf__alpha": uniform(1e-2, 1e-3)
#           }

# Perform randomized search CV to find best hyperparameters
# random_search = RandomizedSearchCV(pclf_NB, param_distributions = params, cv=2,
#                                    verbose = 10, random_state = seed,
#                                    n_iter = 20)
# random_search.fit(X_train, y_train)

# Report results
# report(random_search.cv_results_)

# Assessing model performance on validation set
# y_pred = random_search.predict(X_val)
# print(metrics.classification_report(y_val, y_pred, digits=4))

# Model with rank: 1
# Mean validation score: 0.888 (std: 0.000)
# Parameters: {'clf__alpha': 0.01069475517718527, 'tfidf__use_idf': True, 'vect__binary': False, 'vect__max_df': 0.5, 'vect__ngram_range': (1, 3)}

# Training optimal logistic regression model
pclf_NB.fit(X_train, y_train)
# y_NB_pred_train = pclf_NB.predict(X_train) # required for ensembling method
# y_NB_pred_val = pclf_NB.predict(X_val)
# accuracy_score(y_val, y_NB_pred_val)  #0.8968
y_NB_pred = pclf_NB.predict(X_test)

###########################################################
#                       Ensembling                        #
###########################################################
# The output from the VADER sentiment analyzer, the logistic regression
# model, the Linear SVC model and the Naive Bayes model will be ensembled
# via a stacking method, i.e. the prediction vectors of each will be
# combined to create a 4 column feature matrix. Two different metamodels
# will be used: a voting model and a logistic regression model.

# Combining prediction vectors from each model (including VADER)
# X_combined = np.vstack((y_vader_pred_val, y_LR_pred_val, y_LSVC_pred_val,
#                         y_NB_pred_val)).T
# y_ensemble_pred = np.zeros(len(X_combined))
#
# for i in range(len(X_combined)):
#     if sum(X_combined[i]) >= 3:
#         y_ensemble_pred[i] = 1
#     elif sum(X_combined[i]) <= 1:
#         y_ensemble_pred[i] = 0
#     else:
#         y_ensemble_pred[i] = X_combined[i,2]   # in case of tie, we select
#         # the LSVC prediction as it is the strongest model
#
# accuracy_score(y_val, y_ensemble_pred)  #0.902

# Combining prediction vectors from each model (excluding VADER)
# X_combined = np.vstack((y_LR_pred_val, y_LSVC_pred_val,
#                         y_NB_pred_val)).T
# y_ensemble_pred = np.zeros(len(X_combined))
#
# for i in range(len(X_combined)):
#     if sum(X_combined[i]) >= 2:
#         y_ensemble_pred[i] = 1
#     else:
#         y_ensemble_pred[i] = 0
#
# accuracy_score(y_val, y_ensemble_pred)  #0.900

# Training a logistic regression model - using training set
# X_combined_train = np.vstack((y_vader_pred_train, y_LR_pred_train,
#                          y_LSVC_pred_train,
#                         y_NB_pred_train)).T
# X_combined_val = np.vstack((y_vader_pred_val, y_LR_pred_val,
#                          y_LSVC_pred_val,
#                         y_NB_pred_val)).T
# y_ensemble_pred = np.zeros(len(X_combined_val))
#
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
#     lr = LogisticRegression(C=c)
#     lr.fit(X_combined_train, y_train)
#     print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(
#         X_combined_val))))


# Training a logistic regression metamodel - using training set and including
# interaction terms
# X_combined_train = np.vstack((y_vader_pred_train, y_LR_pred_train,
#                          y_LSVC_pred_train,
#                         y_NB_pred_train, y_vader_pred_train*y_LR_pred_train,
#                          y_vader_pred_train*y_LSVC_pred_train,
#                               y_vader_pred_train*y_NB_pred_train,
#                               y_LR_pred_train*y_LSVC_pred_train,
#                               y_LR_pred_train*y_NB_pred_train,
#                               y_LSVC_pred_train*y_NB_pred_train)).T

# X_combined_val = np.vstack((y_vader_pred_val, y_LR_pred_val,
#                          y_LSVC_pred_val,
#                         y_NB_pred_val, y_vader_pred_val*y_LR_pred_val,
#                          y_vader_pred_val*y_LSVC_pred_val,
#                               y_vader_pred_val*y_NB_pred_val,
#                               y_LR_pred_val*y_LSVC_pred_val,
#                               y_LR_pred_val*y_NB_pred_val,
#                               y_LSVC_pred_val*y_NB_pred_val)).T
# y_ensemble_pred = np.zeros(len(X_combined_val))
#
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
#     lr = LogisticRegression(C=c)
#     lr.fit(X_combined_train, y_train)
#     print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(
#         X_combined_val))))

# Accuracy for C=0.01: 0.9
# Accuracy for C=0.05: 0.9016
# Accuracy for C=0.25: 0.9036
# Accuracy for C=0.5: 0.9036
# Accuracy for C=1: 0.9044

# Training a logistic regression metamodel - excluding VADER (weekest model) 
# and including interaction terms
# X_combined_train = np.vstack((y_LR_pred_train,
#                          y_LSVC_pred_train,
#                         y_NB_pred_train,
#                               y_LR_pred_train*y_LSVC_pred_train,
#                               y_LR_pred_train*y_NB_pred_train,
#                               y_LSVC_pred_train*y_NB_pred_train)).T
#
# X_combined_val = np.vstack((y_LR_pred_val,
#                          y_LSVC_pred_val,
#                         y_NB_pred_val,
#                               y_LR_pred_val*y_LSVC_pred_val,
#                               y_LR_pred_val*y_NB_pred_val,
#                               y_LSVC_pred_val*y_NB_pred_val)).T
# y_ensemble_pred = np.zeros(len(X_combined_val))
#
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
#     lr = LogisticRegression(C=c)
#     lr.fit(X_combined_train, y_train)
#     print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(
#         X_combined_val))))

# Accuracy for C=0.01: 0.9
# Accuracy for C=0.05: 0.9
# Accuracy for C=0.25: 0.9044
# Accuracy for C=0.5: 0.9044
# Accuracy for C=1: 0.9044


# Training a logistic regression metamodel - excluding VADER (weekest model)
# and excluding interaction terms
# X_combined_train = np.vstack((y_LR_pred_train,
#                          y_LSVC_pred_train,
#                         y_NB_pred_train)).T
#
# X_combined_val = np.vstack((y_LR_pred_val,
#                          y_LSVC_pred_val,
#                         y_NB_pred_val)).T
# y_ensemble_pred = np.zeros(len(X_combined_val))
#
# for c in [0.01, 0.05, 0.25, 0.5, 1]:
#     lr = LogisticRegression(C=c)
#     lr.fit(X_combined_train, y_train)
#     print("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(
#         X_combined_val))))

# Accuracy for C=0.01: 0.9
# Accuracy for C=0.05: 0.9
# Accuracy for C=0.25: 0.9
# Accuracy for C=0.5: 0.9
# Accuracy for C=1: 0.9

# X_combined_test = np.vstack((y_vader_pred, y_LR_pred,
#                          y_LSVC_pred,
#                         y_NB_pred, y_vader_pred*y_LR_pred,
#                          y_vader_pred*y_LSVC_pred,
#                               y_vader_pred*y_NB_pred,
#                               y_LR_pred*y_LSVC_pred,
#                               y_LR_pred*y_NB_pred,
#                               y_LSVC_pred*y_NB_pred)).T

# Training best ensembling model
# lr = LogisticRegression(C=1)
# lr.fit(X_combined_train, y_train)


# Predicting categories on test set and saving results as csv, ready for Kaggle
#my_prediction = lr.predict(X_combined_test)
my_prediction = np.array(my_prediction).astype(int)
my_solution = pd.DataFrame(my_prediction, X_test_Id, columns = ['Category'])

# Write Solution to a csv file with the name my_solution.csv
my_solution.to_csv('my_solution.csv', index_label=['Id'])