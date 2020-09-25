# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
movies  = pd.read_csv('movies.csv')
movies = movies.drop(['budget', 'homepage', 'id','popularity', 'production_countries', 'release_date', 'revenue', 'runtime', 'spoken_languages', 'status','vote_average', 'vote_count', 'title', 'overview', 'tagline'], axis=1)
features = ['genres','keywords','original_language','original_title','production_companies']
for feature in features:
    movies[feature].fillna('')


# %%
movies.head()


# %%
import string
import re
def remove_punc(row):
   return ''.join([w.lower() for w in row if w not in string.punctuation])

def remove_labels(row):
   return ' '.join([w for w in row.split() if not re.search('^id$|name$|[0-9]', w)])

movies['genres'] = movies['genres'].apply(remove_punc)
movies['genres'] = movies['genres'].apply(remove_labels)
movies['keywords'] = movies['keywords'].apply(remove_punc)
movies['keywords'] = movies['keywords'].apply(remove_labels)
movies['production_companies'] = movies['production_companies'].apply(remove_punc)
movies['production_companies'] = movies['production_companies'].apply(remove_labels)
movies['original_title'] = movies['original_title'].apply(remove_punc)


# %%
movies.head()


# %%
def combine_features(r):
    return r['genres'] + r['keywords'] + r['original_language'] + r['original_title']  + r['production_companies']
movies['combined_features'] = movies.apply(combine_features, axis=1)


# %%
movies.head()


# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
count_matrix = CountVectorizer().fit_transform(movies['combined_features'])


# %%
cosine_sim = cosine_similarity(count_matrix)


# %%
def get_title_from_index(index):
    return movies[movies.index == index].original_title[index]

def get_index_from_title(title):
    return movies[movies.original_title == title].index.values[0]


# %%
movie_user_likes = "avengers age of ultron"
# get_index_from_title(movie_user_likes)
movie_index = get_index_from_title(movie_user_likes)
similar_movies =  list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

i=0
print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>=5:
        break




