import os
import ntpath
import glob
import nltk

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
    for i in range(len(wordlist)):
        matrix.append([0, 0])
    for word in wordlist:
        for poscom in positive:
            for w in poscom['text']:
                if word[0] == w:
                    matrix[wordlist.index(word)][0] += 1
        for negcom in negative:
            for w in negcom['text']:
                if word[0] == w:
                    matrix[wordlist.index(word)][1] += 1
    bayes = []
    for b in matrix:
        bayes.append([(b[0]/(b[0]+b[1])), (b[1]/(b[0]+b[1]))])
    return bayes

print(bayesprobmatrix(poscomments, negcomments, word_160pos))
