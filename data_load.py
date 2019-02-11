import os
import ntpath
import glob
import nltk
import numpy as np
import math

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


"""J'ai repris mes anciennes fonctions de traitement de texte pour pouvoir 
avancer le reste, on trouvera mieux avec Lemmatize etc. après"""
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


def bayesprobmatrix(positive, negative, wordlist):
    matrix = []
    num_examples_pos = len(positive)
    num_examples_neg = len(negative)
    word_found = False  #bool to prevent counting words more than once in
    # each comment
    for i in range(len(wordlist)):
        matrix.append([0, 0])
    for word in wordlist:
        for poscom in positive:
            for w in poscom['text']:
                if word[0] == w and word_found is False:
                    word_found = True
                    matrix[wordlist.index(word)][0] += 1
            word_found = False
        for negcom in negative:
            for w in negcom['text']:
                if word[0] == w and word_found is False:
                    word_found = True
                    matrix[wordlist.index(word)][1] += 1
            word_found = False
    bayes = []
    for b in matrix:    #adding Laplace smoothing
        bayes.append([(b[0]+1)/(num_examples_neg+2),
                      (b[1]+1)/(num_examples_pos+2)])
    return bayes

def NaiveBayes(X_train, y_train, X_new):
    # Inputs:
    # -X_train, a feature matrix (size n x m), which is composed of
    # binary variables (0, 1);
    # -y_train (n), target vector composed of binary variables (0, 1).
    # -X_new (n x m)

    # Define # of instances and features in X_train for future use
    n = len(X_train)    # number of instances
    m = len(X_train[0]) # number of features

    # Instantiate theta. e.g. theta_j1 corresponds to the Pr (x_j =1|y=1)
    theta = np.zeros((m, 2))

    # Looping over all instances in feature matrix, for each feature, computing
    # the prior probability Pr(x_j | y=1) and Pr(x_j | y=0)
    for j in range(0,m):
        for i in range(0,n):
            if X_train[i][j] == 1 and y_train[i] == 1:
                theta[j,1] += 1
            elif X_train[i][j] == 1 and y_train[i] == 0:
                theta[j,0] += 1
        # Computing priors and implementing Laplace smoothing
        theta[j, 1] = (theta[j, 1] + 1) / (sum(y_train) + 2)
        theta[j, 0] = (theta[j, 0] + 1) / (n - sum(y_train) + 2)
    theta_1 = sum(y_train) / n
    theta_0 = 1 - theta_1

    # Now that the Bernoulli Naive Bayes is trained, we can use it to predict
    # classes on the new instances

    # Define # of instances and features in X_train for future use
    n_new = len(X_train)    # number of instances
    m_new = len(X_train[0]) # number of features

    for i in range(0,n_new):
        a_0 = 0
        a_1 = 0
        for j in range(0,m_new):
            a_0 = a_0 + X_new[i, j]*math.log(theta[j, 0]) \
                  + (1 - X_new[i, j]) * math.log(1-theta[j, 0])
            a_1 = a_1 + X_new[i, j]*math.log(theta[j, 1]) \
                  + (1 - X_new[i, j]) * math.log(1-theta[j, 1])
        a_0 = a_0 + math.log(theta_0)
        a_1 = a_1 + math.log(theta_1)

        # for each new instance, we pick the highest score between a_0 or
        # a_1 to predict the class

        y_predict = []
        if a_1 >= a_0:
            y_predict.append(1)
        else:
            y_predict.append(0)

    return y_predict

theta = bayesprobmatrix(poscomments, negcomments, word_160pos)
#print(bayesprobmatrix(poscomments, negcomments, word_160pos))
