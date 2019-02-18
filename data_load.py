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
from sklearn.naive_bayes import BernoulliNB

def extract_comments(folder):
    # Set folder:
    fold = "train/" + folder + "/"

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


def get_training_data(folder):
    # Set folder:
    fold = folder + "/pos" + "/"

    # Get filepaths for all files which end with ".txt"
    filepaths = glob.glob(os.path.join(fold, '*.txt'))  # Mac
    #filepaths = glob.glob(ntpath.join(fold, '*.txt'))   # Windows

    # Create an empty list for collecting the comments
    commentlist = []

    # iterate for each file path in the list
    for fp in filepaths:
        comment = {}
        # Open the file in read mode
        with open(fp, 'r', encoding="utf8") as f:
            # Read the first line of the file
            comment = f.read().lower()
        commentlist.append(comment)
            # Append the first line into the headers-list

    # Creating feature array for positive comments
    X = np.asarray(commentlist)

    # Creating target vector of ones for the positive comments
    y = np.ones(len(commentlist))

    # Set folder:
    fold = "train/" + "neg" + "/"

    # Get filepaths for all files which end with ".txt"
    filepaths = glob.glob(os.path.join(fold, '*.txt'))  # Mac
    # filepaths = glob.glob(ntpath.join(fold, '*.txt'))   # Windows

    # Create an empty list for collecting the comments
    commentlist2 = []

    # iterate for each file path in the list
    for fp in filepaths:
        comment = {}
        # Open the file in read mode
        with open(fp, 'r', encoding="utf8") as f:
            # Read the first line of the file
            comment= f.read().lower()
        commentlist2.append(comment)
        # Append the first line into the headers-list

    # Extending X feature array with negative comments
    X = np.concatenate([X, np.asarray(commentlist2)])

    # Extending target vector with zeros for all negative comments
    y = np.concatenate([y, np.zeros(len(commentlist2))])

    return X, y


def get_test_data(folder):
    # Set folder:
    fold = folder + "/"

    # Get filepaths for all files which end with ".txt"
    filepaths = glob.glob(os.path.join(fold, '*.txt'))  # Mac
    #filepaths = glob.glob(ntpath.join(fold, '*.txt'))   # Windows

    # Create an empty list for collecting the comments
    commentlist = []
    IDlist = []

    # iterate for each file path in the list
    for fp in filepaths:
        comment = {}
        ID = {}
        ID = fp[:-4]
        ID = ID[5:]
        # Open the file in read mode
        with open(fp, 'r', encoding="utf8") as f:
            # Read the first line of the file
            comment = f.read().lower()
        commentlist.append(comment)
        IDlist.append(int(ID))
            # Append the first line into the headers-list

    # Creating feature array
    X = np.asarray(commentlist)
    X_id = np.asarray(IDlist)

    return X, X_id

# From: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
