import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from data_load import extract_comments
from data_load import WordListing


def bern_nb_training_set(positive, negative, wordlist):
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


def bern_nb_train(X_train, y_train):
    # Inputs:
    # -X_train, a feature matrix (size n x m), which is composed of
    # binary variables (0, 1);
    # -y_train (n), target vector composed of binary variables (0, 1).

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

    return theta_mat, theta_0, theta_1


def bern_nb_predict(X_new, theta_mat, theta_0, theta_1):
    # Define # of instances and features in X_train for future use
    n_new = len(X_new)  # number of instances
    m_new = len(X_new[0])  # number of features

    # Instantiating the y_predict vector
    y_predict = np.zeros(n_new)

    for i in range(0, n_new):
        sum=0
        for j in range(0, m_new):
            sum = sum + (X_new[i,j] * math.log(theta_mat[j,1]/theta_mat[j,
                                                                         0])
                          + (1-X_new[i,j]) * math.log((1-theta_mat[j,
                                                                 1])/(
                            1-theta_mat[j,0])))
        delta = math.log(theta_1 / (1 - theta_1)) + sum

        if delta >= 0:
            y_predict[i] = 1

    return y_predict


# After going trough all the files, print the list of headers
negcomments = extract_comments('neg')
poscomments = extract_comments('pos')
print(negcomments[10000])
print(poscomments[10000])

# Creating list of popular words
word_160neg = WordListing(negcomments, 160)
word_160pos = WordListing(poscomments, 160)
word_all = WordListing(negcomments+poscomments, 1000)
print(word_160neg)
print(word_160pos)

# Creating training feature matrix and target vector
X, y = bern_nb_training_set(poscomments, negcomments, word_all)

# Splitting them so as to have validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.9, \
                                                  test_size=0.1,
                                                  random_state=123)

# Fitting a Bernoulli NB model and using it to predict labels on validation set
theta_mat_val, theta_0_val, theta_1_val = bern_nb_train(X_train, y_train)
y_predict = bern_nb_predict(X_val, theta_mat_val, theta_0_val, theta_1_val)
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

# Comparison Bernouilli NB
ber = BernoulliNB().fit(X_train, y_train)
y_pred_ber= ber.predict(X_val)
score_scikit3 = accuracy_score(y_val, y_pred_ber)
print("Bernoulli NB (Scikit) accuracy: ", score_scikit3)
