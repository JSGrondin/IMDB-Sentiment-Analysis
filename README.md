# IMDB Sentiment Analysis
This mini-project was undertaken as part of COMP-551 at McGill. 

The following was accomplished: 
- A Bernoulli Naive Bayes model was created from scratch. 
- Experiments were run using different classifiers (MNB, SVM and Log. Reg) to predict the sentiment in IMDB reviews. 

The following python scripts were used: 

-data_load.py : extracting comments from all .txt files

-textprocessing.py: removing special characters, stopwords, urls, emails, lemmatize or stemming

-NaiveBayes.py: bernoulli naive model implemented from scratch

-pipeline.py : main script calling functions from previous listed files, with addition of sklearn functions. An ensembling method was also used to combine the results from the various classifiers. 

See the writeup.pdf for the details of the work and results. 
