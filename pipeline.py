# Importing relevant packages
import pandas as pd
import numpy as np
from data_load import get_training_data
from data_load import get_test_data
from data_load import report
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

# Extracting raw features (X) and targets(y)
X_train, y_train = get_training_data('train')
X_test, X_test_Id = get_test_data('test')

# Setting Seed
seed = 123

# Splitting them so as to have validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                  train_size=0.9, \
                                                  test_size=0.1,
                                                  random_state=seed)

# Building Multinomial NB pipeline
pclf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('norm', Normalizer()),
    ('clf', MultinomialNB()),
])

params = {"vect__ngram_range": [(1,1),(1,2),(2,2)],
          "tfidf__use_idf": [True, False],
          "clf__alpha": uniform(1e-2, 1e-3)}

# Perform randomized search CV to find best hyperparameters
random_search = RandomizedSearchCV(pclf, param_distributions = params, cv=2,
                                   verbose = 10, random_state = seed,
                                   n_iter = 5)
random_search.fit(X_train, y_train)

# Report results
report(random_search.cv_results_)

# Assessing model performance on validation set
y_pred = random_search.predict(X_val)
print(metrics.classification_report(y_val, y_pred))

# Predicting categories on test set and saving results as csv, ready for Kaggle
my_prediction = random_search.predict(X_test)
my_prediction = np.array(my_prediction).astype(int)
my_solution = pd.DataFrame(my_prediction, X_test_Id, columns = ['Category'])

# Write Solution to a csv file with the name my_solution.csv
my_solution.to_csv('my_solution.csv', index_label=['Id'])