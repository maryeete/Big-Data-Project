import pandas as pd
import mysql.connector
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import string
import nltk
nltk.download('all')
#nltk.download('stopwords')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_tweets(host,username, password, database, query):
    mydata = mysql.connector.connect(
    host=host,
    user=username,
    password=password,
    database=database)

    cursor = mydata.cursor()

    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    mydata.close()

    data = {'tweet_id': [], 'tweet_text': [], 'event_label':[]}
    for row in rows:
        data['tweet_id'].append(row[0])  
        data['tweet_text'].append(row[1]) 
        data['event_label'].append(row[2])
    return pd.DataFrame(data)



# Preprocessing function to clean and tokenize tweets
def preprocess_tweet(tweet):
    tweet = tweet.lower()  # Convert to lowercase
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = word_tokenize(tweet)  # Tokenize
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    return ' '.join(tokens)


def train_model(train_tweets):
    train_tweets['clean_text'] = train_tweets['tweet_text'].apply(preprocess_tweet)
    print(train_tweets)
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_tweets['clean_text'])

    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, train_tweets['event_label'])

    return nb_classifier, tfidf_vectorizer


def test_model(test_tweets, classifier, vectorizer):
    test_tweets['clean_text'] = test_tweets['tweet_text'].apply(preprocess_tweet)
    print(test_tweets)
    X_test_tfidf = vectorizer.transform(test_tweets['clean_text'])

    y_pred = classifier.predict(X_test_tfidf)

    test_tweets['predicted_event_label'] = y_pred
    return test_tweets


#Database Usage
host = 'localhost'
username = 'root'
password = '@MySeniorProJecT21'
database = 'big_data'

query = 'Select * from tweets;'

tweets_df = load_tweets(host, username, password, database, query)
print(tweets_df)
print(tweets_df.columns)

# Split the dataset into training (first 150 rows) and testing (rest of the rows)
train_tweets = tweets_df.iloc[:150]
test_tweets = tweets_df.iloc[150:]

# Train the model
nb_classifier, tfidf_vectorizer = train_model(train_tweets)

# Test the model
predicted_tweets = test_model(test_tweets, nb_classifier, tfidf_vectorizer)

# Print the predicted tweets
print(predicted_tweets)
predicted_tweets.to_csv('Predicted_Tweets.csv', index=False)
print("Predicted tweets exported to Predicted_Tweets.csv")
