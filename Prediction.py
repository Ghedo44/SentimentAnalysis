import joblib
import pickle
import praw
import pandas as pd
from pandas import ExcelWriter
import datetime as dt
import matplotlib.pyplot as plt
from collections import Counter
import string

from tqdm import tqdm

# Prendo i dati da reddit
reddit = praw.Reddit(client_id='TIb8sBV_RGzAZA',
                     client_secret='u0owlsQMwlikuunEXOnAh_JLq51ocQ',
                     user_agent='Sentiment Analisys',
                     username='SentimentAnalisys',
                     password='ciao1234')

posts = []

source = 'wallstreetbets'  # subreddit da analizzare. Con 'all' prende dati da tutti i subreddit
keyword = 'GME'
limit = 10

subreddit = reddit.subreddit(source)

pbar = tqdm(total=limit)
pbar.set_description("Collecting posts")
for post in subreddit.search(keyword, limit=limit):
    posts.append([post.title, post.score, post.id, 'https://www.reddit.com'+post.permalink,
                  post.subreddit, post.url, post.num_comments, post.selftext, dt.datetime.fromtimestamp(post.created)])
    pbar.set_description("Collecting posts")
    pbar.update(1)
pbar.close()

posts = pd.DataFrame(posts, columns=['title', 'score', 'id', 'permalink',
                                     'subreddit', 'url', 'num_comments', 'body', 'created'])


comments = set()
x = 0
pbar = tqdm(total=len(posts['id']))
pbar.set_description("Collecting comments from posts")
while x < len(posts['id']):
    submission = reddit.submission(id=posts['id'][x])
    x = x + 1

    # Prendo i commenti più popolari
    submission.comments.replace_more(limit=0)
    for top_level_comment in submission.comments:
        comments.add(top_level_comment.body)

    pbar.set_description("Collecting comments from posts")
    pbar.update(1)
pbar.close()


# Metto i commenti in un df
df1 = pd.DataFrame(comments, columns=['Comments'])

# carico il vectorizer e vettorizzo i commenti
loaded_vectorizer = pickle.load(open('vectorizer.pickle', 'rb'))
df1['Vect'] = loaded_vectorizer.transform(df1['Comments'])

# Carico il modello addestrato
knn_from_joblib = joblib.load('reddit.pkl')


# Utilizzo il modello per fare le predizioni
pbar = tqdm(total=len(df1['Vect']))
for line in df1['Vect']:
    df1['Predictions'] = knn_from_joblib.predict(line)
    pbar.set_description("Making predictions")
    pbar.update(1)
pbar.close()

# Calcolo numero commenti positivi e negativi
pos = 0
neg = 0

for prediction in df1['Predictions']:
    if prediction > 0:
        pos = pos + 1
    else:
        neg = neg + 1

print("\nTotal Positive = ", pos)
print("Total Negative = ", neg)


# Conto le parole più ricorrenti
data_set = df1['Comments']

split = []
for line in data_set:
    split.append(line.split())

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "dont", "get", "", "day", "cover",
              "\u200d", "tomorrow", "im", "us"]

final_words = []
for line in split:
    for word in line:
        word = word.lower().translate(str.maketrans('', '', string.punctuation))
        if word not in stop_words:
            final_words.append(word)

Counter = Counter(final_words)

most_occur = Counter.most_common(10)

print("\n")
print(most_occur)

df2 = pd.DataFrame(most_occur, columns=['Comments', 'Number'])


# Data attuale da mettere nel nome del grafico e Excel
now = str(dt.datetime.now())

# Salvo il dataframe in Excel
writer = ExcelWriter('Subreddit=' + source + '_Keyword=' + keyword + '_Date=' + now[:10] + '.xlsx')
df1.to_excel(writer, 'sheet1')
df2.to_excel(writer, 'sheet2')
writer.save()

# Creo i grafici
fig, axl = plt.subplots()
axl.bar(df2['Comments'], df2['Number'])
fig.autofmt_xdate()
plt.savefig('Subreddit=' + source + '_Keyword=' + keyword + '_Date=' + now[:10] + '_RecurrentWords' + '.png')
plt.show()

labels = 'Positive', 'Negative'
sizes = [pos, neg]
colors = ['lightgreen', 'lightcoral']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.savefig('Subreddit=' + source + '_Keyword=' + keyword + '_Date=' + now[:10] + '_Sentiment' + '.png')
plt.show()
