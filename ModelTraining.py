import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pickle

# Carico il dataset
df = pd.read_csv('/content/drive/MyDrive/SentimentAnalisys/stock_data.csv')

X = df['Text']
y = df['Sentiment'] #prova

vect = CountVectorizer()  # ngram_range=(1, 2)
X = vect.fit_transform(X)

# Salvo il vectorizer
vec_file = '/content/drive/MyDrive/SentimentAnalisys/models/vectorizer.pickle'
pickle.dump(vect, open(vec_file, 'wb'))

# Divido il Dataset in Train e Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# importo il modello KNN
knn = KNN(n_neighbors=5)

# Addestro il modello
knn.fit(X_train, y_train)

# Predizioni
p_train = knn.predict(X_train)
p_test = knn.predict(X_test)

# Accuratezza
acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print(f'Train accuracy score: {acc_train}, test accuracy score: {acc_test}')


# Salvo il modello
joblib.dump(knn, '/content/drive/MyDrive/SentimentAnalisys/models/reddit.pkl')
