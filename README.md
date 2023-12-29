# MOVIE-RECOMMENDER

### Movie Recommendation System Using Cosine Similarity

MOVIE-RECOMMENDER ,relying on historical user interactions, provides movie recommendations by emphasizing uniqueness. It suggests movies to users if the likelihood of the recommendation is higher than the user's past preferences for similar content. In essence, MoRe offers suggestions that may not have been popular among other users but could be appreciated by an individual user based on their distinct preferences.

## Table of Content
- [Introduction to Recommendation System](#introduction-to-recommendation-system)
- [Cosine Similarity](#cosine-similarity)
- [Code](#code)

#### Introduction to Recommendation System
Recommendation systems are designed to suggest items to users by considering various factors. These systems predict items that users are inclined to buy or have an interest in. Major corporations like Google, Amazon, and Netflix utilize recommendation systems to assist users in making purchases, whether it's products or movies. The system suggests items either based on users' past activities, known as Content-Based Filtering, or by considering the preferences of other users with similar tastes, referred to as Collaborative-Based Filtering.
#### Cosine Similarity 
Cosine similarity serves as a metric for assessing the similarity between two items. It employs mathematical calculations to determine the cosine of the angle formed by two vectors projected in a multidimensional space. The advantage of cosine similarity becomes evident when similar documents are situated far apart in terms of Euclidean distance (document size). In such cases, they may still be closely aligned in orientation. Essentially, the smaller the angle, the greater the cosine similarity between the items.```
```
1 - cosine-similarity = cosine-distance
```



#### Dataset
Download the dataset from [here](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

##### Importing the important libraries

```python3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import nltk
from nltk.stem.porter import PorterStemmer
```
##### Loading the dataset and converting it into dataframe using pandas

```python3
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')
```

##### Merge the DataFrames
merging on basis of title

```python3
movies=movies.merge(credits,on='title')
```

##### Useful Features 
```python3
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
```
##### Dropping null values 
```python3
movies.dropna(inplace =True)
```

##### Converting Genres and keywords feature values
```python3
import ast
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
```
##### Converting Cast feature values
```python3
import ast
def convert3(text):
    L = []
    counter=0
    for i in ast.literal_eval(text):
        if counter != 3:
            L.append(i['name']) 
            counter+=1
        else:
            break
    return L
movies['cast']=movies['cast'].apply(convert3)
```
##### Converting Crew feature values
```python3
import ast
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name']) 
            break
    return L 
movies['crew']=movies['crew'].apply(fetch_director)
```


##### Splitting overview values
```python3
movies['overview']=movies['overview'].apply(lambda x:x.split())
```
##### Removing space between words
converted all columns to list of words
get unique words and remove space between words like between samworthington to stop it from breaking into two different words
science fiction to sciencefiction
```python3
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['overview']=movies['overview'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

```
##### Generating similar movies matrix
create new column and make them as tags(concatenate all the columns into one i.e. tag)
```python3
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
```
##### Creating a New DataFrame and doing pre-processing
create new column and make them as tags(concatenate all the columns into one i.e. tag)
```python3
new_df=movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
```
##### Using Count-Vectorizer
```python3
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
```

##### Cosine-Distance
```python3
similarity=cosine_similarity(vectors)
```
##### Predicting
```python3
def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
recommend('Avatar')
```


![Final Output](https://github.com/DivyaArora873/movie-recommender-system/blob/main/Screenshot%202023-12-29%20151747.png)
