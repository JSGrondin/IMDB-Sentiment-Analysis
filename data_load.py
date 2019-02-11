import os
import glob
#import nltk

def extract_comments(folder):
    # Set folder:
    fold = "train/"+ folder +"/"

    # Get filepaths for all files which end with ".txt"
    filepaths = glob.glob(os.path.join(fold, '*.txt'))


    # Create an empty list for collecting the comments
    commentlist = []

    # iterate for each file path in the list
    for fp in filepaths:
        comment = {}
        ID = fp[10:]
        comment['ID'] = ID[:-6]
        # Open the file in read mode
        with open(fp, 'r') as f:
            # Read the first line of the file
            comment['text'] = f.read().lower().split()
        commentlist.append(comment)
            # Append the first line into the headers-list

    return commentlist

    # After going trough all the files, print the list of headers
negcomments = extract_comments('neg')
poscomments = extract_comments('pos')
print(negcomments[10000:10001])
print(poscomments[10000])
