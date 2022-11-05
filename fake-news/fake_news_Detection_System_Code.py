import numpy as np  # for array
import pandas as pd  # for datadrame
# for searching the text into the tarin data
import re
from nltk.corpus import stopwords  # Remove useless words
# it give the rot words for a particular word
from nltk.stem.porter import PorterStemmer
# Convert text into Feature vector
from sklearn.feature_extraction.text import TfidfVectorizer
# split the actual data into taxt and training data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # LogisticRegrassion
from sklearn.metrics import accuracy_score  # for accuracy for model

import nltk
nltk.download('stopwords')

# printing the stop words in english
print(stopwords.words('english'))

# DATA PRE PROCESSING
# Loading the dataset to a pandas Dataframe

news_dataset = pd.read_csv(
    '/content/train.csv', engine="python", error_bad_lines=False, encoding='utf-8')

news_dataset

news_dataset.tail(500)

# counting the number of missing value in the dataset
news_dataset.isnull().sum()

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

news_dataset.isnull().sum()

# Merging the author name and news title
news_dataset['content'] = news_dataset['author'] + ' '+news_dataset['title']

print(news_dataset['content'])

# seprating the data label
x = news_dataset.drop(columns='label', axis=1)
y = news_dataset['label']
print(x)
print(y)

"""stemming:
it is a process of reducing a word to its root word
ex-> Actor,Actress,Acting->Root Word is Act
"""

port_stem = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(
        word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

# separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)

print(Y)

Y.shape

# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

print(X)

# Splitting the dataset to training & test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)

# Training the Model: Logistic Regression
model = LogisticRegression()

model.fit(X_train, Y_train)

# Evaluation

# accuracy score

# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the training data : ', test_data_accuracy)

"""Making a Predictive System

"""

X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0] == 0):
    print('The news is Real')
else:
    print('The news is Fake')

print(Y_test[3])
