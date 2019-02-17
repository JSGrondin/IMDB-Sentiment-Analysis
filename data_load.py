import os
import ntpath
import glob
import nltk
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

def extract_comments(folder):
    # Set folder:
    fold = "train/"+ folder +"/"

    # Get filepaths for all files which end with ".txt"
    filepaths = glob.glob(os.path.join(fold, '*.txt'))  # Mac
    #filepaths = glob.glob(ntpath.join(fold, '*.txt'))   # Windows

    # Create an empty list for collecting the comments
    commentlist = []

    # iterate for each file path in the list
    for fp in filepaths:
        comment = {}
        ID = fp[10:]
        comment['ID'] = ID[:-6]
        # Open the file in read mode
        with open(fp, 'r', encoding="utf8") as f:
            # Read the first line of the file
            comment['text'] = f.read().lower().split()
        commentlist.append(comment)
            # Append the first line into the headers-list

    return commentlist


# After going trough all the files, print the list of headers
negcomments = extract_comments('neg')
poscomments = extract_comments('pos')
print(negcomments[10000])
print(poscomments[10000])


def get_word_count(words, words_set):
    word_count = {w: 0 for w in words_set}
    for w in words:
        word_count[w] += 1
    return word_count


# Create list of all the words we have in all the comments
def WordListing(data, nb_words, start=0):
    words = []
    for comment in data:
        for word in comment["text"]:
            words.append(word)

    # Transform into set to eliminate duplicates
    words_set = set(words)

    # Count of occurrences of every word
    word_count = get_word_count(words, words_set)

    # Create list of 160 most occurrent words
    word_list =[]
    for w in sorted(word_count, key=word_count.get, reverse=True)[start:nb_words]:
        word_list.append([w, word_count[w]])

    return word_list

word_160neg = WordListing(negcomments, 160)
word_160pos = WordListing(poscomments, 160)
print(word_160neg)
print(word_160pos)


def bayes_training_set(positive, negative, wordlist):
    # This function is used to create the feature matrix (X) and target vector
    # (y) for the training set. All features and targets are binary values,
    # i.e. can only take values 0 or 1. This is a requirement for the
    # Bernoulli NB model.

    num_examples_pos = len(positive)
    num_examples_neg = len(negative)
    num_features = len(wordlist)

    # instantiating X and y, using numpy arrays
    X = np.zeros((num_examples_neg + num_examples_pos, num_features))
    y = np.ones(num_examples_neg)
    y = np.concatenate((y, np.zeros(num_examples_pos)))

    for j, word in enumerate(wordlist):
        for i, poscom in enumerate(positive):
            for w in poscom['text']:
                if word[0] == w:
                    X[i, j] = 1
        for i, negcom in enumerate(negative):
            for w in negcom['text']:
                if word[0] == w:
                    X[num_examples_pos + i, j] = 1
    return X, y


def bern_naive_bayes(X_train, y_train):
    # Inputs:
    # -X_train, a feature matrix (size n x m), which is composed of
    # binary variables (0, 1);
    # -y_train (n), target vector composed of binary variables (0, 1).
    # -X_new is a feature matrix (size n2 x m2), which is also composed of
    # binary variables (0, 1) and for which we want to predict labels using
    # the newly trained NB model

    # Define # of instances and features in X_train for future use
    n = len(X_train)    # number of instances
    m = len(X_train[0]) # number of features

    # Instantiate theta. e.g. theta_j1 corresponds to the Pr (x_j =1|y=1)
    theta_mat = np.zeros((m, 2))

    # Looping over all instances in feature matrix, for each feature, computing
    # the prior probability Pr(x_j | y=1) and Pr(x_j | y=0). This is
    # equivalent to training the Bernoulli NB model
    for j in range(0, m):
        for i in range(0, n):
            if X_train[i, j] == 1 and y_train[i] == 1:
                theta_mat[j, 1] += 1
            elif X_train[i, j] == 1 and y_train[i] == 0:
                theta_mat[j,0] += 1
        # Computing priors and implementing Laplace smoothing
        theta_mat[j, 1] = (theta_mat[j, 1] + 1) / (sum(y_train) + 2)
        theta_mat[j, 0] = (theta_mat[j, 0] + 1) / (n - sum(y_train) + 2)
    theta_1 = sum(y_train) / n
    theta_0 = 1 - theta_1

    # Now that the Bernoulli Naive Bayes is trained, we can use it to predict
    # classes on the new instances (i.e. X_val or X_test)
    return theta_mat, theta_0, theta_1


def predict(X_new, theta_mat, theta_0, theta_1):
    # Define # of instances and features in X_train for future use
    n_new = len(X_new)  # number of instances
    m_new = len(X_new[0])  # number of features

    # Instantiating the y_predict vector
    y_predict = np.zeros(n_new)

    for i in range(0, n_new):
        a_0 = 0
        a_1 = 0
        for j in range(0, m_new):
            a_0 = a_0 + X_new[i, j] * math.log(theta_mat[j, 0]) \
                  + (1 - X_new[i, j]) * math.log(1-theta_mat[j, 0])

            a_1 = a_1 + X_new[i, j] * math.log(theta_mat[j, 1]) \
                  + (1 - X_new[i, j]) * math.log(1-theta_mat[j, 1])

        a_0 = a_0 + math.log(theta_0)
        a_1 = a_1 + math.log(theta_1)

        # for each new instance, we pick the highest score between a_0 or
        # a_1 to predict the class

        if a_1 >= a_0:
            y_predict[i] = 1

    return y_predict


# Creating training feature matrix and target vector
X, y = bayes_training_set(poscomments, negcomments, word_160pos)

# Splitting them so as to have validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, \
                                                  test_size=0.2,
                                                  random_state=123)

# Fitting a Bernoulli NB model and using it to predict labels on validation set
theta_mat_val, theta_0_val, theta_1_val = bern_naive_bayes(X_train, y_train)
y_predict = predict(X_val, theta_mat_val, theta_0_val, theta_1_val)
score = accuracy_score(y_val, y_predict)
print("NB accuracy: ", score)


# Comparison : Multinomial NB
clf = MultinomialNB().fit(X_train, y_train)
scikit_predict = clf.predict(X_val)
score_scikit = accuracy_score(y_val, scikit_predict)
print("NB scikit accuracy: ", score_scikit)


# Comparison Gaussian NB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_val)
score_scikit2 = accuracy_score(y_val, y_pred)
print("NB scikit accuracy: ", score_scikit2)

print("Number of mislabeled points out of a total %d points : %d",
      (X_train.shape[0],(y_val != y_pred).sum()))
