import pandas as pd
import sklearn
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import nltk.data
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

moviedir=r"C:\file_directory"

# loading all files as training data.
movie_train = load_files(moviedir)
filtered_words = [word for word in movie_train if word not in stopwords.words('english')]



# first file is in "neg" folder
#print(movie_train.filenames[0])


# initialize movie_vector object, and then turn movie train data into a vector
movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize)         # use all 25K words. 82.2% acc.

# movie_vec = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize, max_features = 3000) # use top 3000 words only. 78.5% acc.
movie_counts = movie_vec.fit_transform(movie_train.data)


#2,000 documents, 25K unique terms.
test3 = movie_counts.shape
print(test3)

# Convert raw frequency counts into TF-IDF values
tfidf_transformer = TfidfTransformer()
movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

# Same dimensions, now with tf-idf values instead of raw frequency counts
movie_tfidf.shape

# Same dimensions, now with tf-idf values instead of raw frequency counts
movie_tfidf.shape

# Now ready to build a classifier.

# from sklearn.cross_validation import train_test_split  # deprecated in 0.18

docs_train, docs_test, y_train, y_test = train_test_split(
    movie_tfidf, movie_train.target, test_size = 0.20, random_state = 12)

# Train a Random forest Classifier
classifier = RandomForestClassifier(n_estimators=1000)
clf=classifier.fit(docs_train, y_train)
#clf = MultinomialNB().fit(docs_train, y_train)

# Predicting the Test set results, find accuracy
y_pred = clf.predict(docs_test)
sklearn.metrics.accuracy_score(y_test, y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#Trying classifier on fake movie reviews
# very short and fake movie reviews
reviews_new = ['This movie was excellent',
               'Absolute joy ride',
               'This was certainly a movie',
               'Two thumbs up',
               'I fell asleep halfway through',
               'We cant wait for the sequel',
               'I cannot recommend this highly enough']
reviews_new_counts = movie_vec.transform(reviews_new)
reviews_new_tfidf = tfidf_transformer.transform(reviews_new_counts)

# have classifier make a prediction
pred = clf.predict(reviews_new_tfidf)


for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie_train.target_names[category]))

