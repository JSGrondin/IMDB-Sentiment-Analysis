import data_load as dl
import re as regex
import nltk
import numpy as np
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

def char_preprocessing(X):
    REPLACE_NO_SPACE = regex.compile(
        "(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\*)|(\^)|(\+)|(\=)|(\{)|(\})|(\&)")
    REPLACE_WITH_SPACE = regex.compile("(<br\s*/><br\s*/>)|(><br\s*>)(\-)")
    REPLACE_NO_SPACE2 = regex.compile("(\<)|(\>)")
    X = [REPLACE_NO_SPACE.sub("", line) for line in X]
    X = [REPLACE_WITH_SPACE.sub(" ", line) for line in X]
    X = [REPLACE_NO_SPACE2.sub("", line) for line in X]

    X = np.asarray(X)
    return X


def remove_urls(X):
    regexp = r'http.?://[^\s]+[\s]?'

    for i, text in enumerate(X):
        X[i] = regex.sub(regexp, '', text)

    return X


def remove_emails(X):
    regexp = r'\S*@\S*\s?'

    for i, text in enumerate(X):
        X[i] = regex.sub(regexp, '', text)

    return X

def special_characters(X):

    remove = '[()",.!@#$%?&*-^+={}><;:]'
    for i, text in enumerate(X):
        X[i] = regex.sub(remove, "", text)

    return X


def lemmatization_stem_and_stopwords(X, remove_stopwords=True,
                                     lemmatize=True, stem=True):
    lemma = WordNetLemmatizer()
    #snow = nltk.stem.SnowballStemmer('english')
    snow = PorterStemmer()
#    stopwords = nltk.corpus.stopwords.words("english")
    stopwords = ['in', 'of', 'at', 'a', 'the']
    whitelist = ["n't", "not"]
    for i, comment in enumerate(X):
        textlem = ''

        for word in comment.split():

            if (remove_stopwords == True and word in stopwords and word not in
                    whitelist):
                a=1 # dummy expression

            else:
                if lemmatize == True:
                    lem = lemma.lemmatize(word)

                else:
                    lem = word

                if stem == True:
                    sno = snow.stem(lem)

                else:
                    sno = lem

                textlem += sno + ' '

        X[i] = textlem

    return X


if __name__ == '__main__':

    X, y = dl.get_training_data()

    print(X[3])

    K = special_characters(X)

    print(K[3])

    J = lemmatization_stem_and_stopwords(K)

    print(J[3])
