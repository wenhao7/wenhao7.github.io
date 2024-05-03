---
layout: post
title:  "Exploring Anime Recommendation System Approaches"
date:   2024-05-03 13:57:09 +0800
category: [data_analysis, machine_learning, visualization, misc]
tag: [numpy, pandas, seaborn, statistics scipy, nltk, scikit-learn, multiprocessing, math, matplotlib, nlp, recommendation, featured, dimensionality_reduction]
summary: "In this notebook we will be exploring recommendation systems using 3 different approaches, applying them to data we have scraped from a popular anime database/community website."
image: /images/banners/mal_rec.png
---

1. [Introduction](#1)
2. [Data](#2)
3. [Baseline Model & Evaluation](#3)
4. [Content Based Filtering](#4)
5. [Collaborative Filtering](#5)
6. [Hybrid Model](#6)
7. [Conclusion](#7)

<a id='1'></a>
## 1. Introduction
In this notebook we will be exploring recommendation systems using 3 different approaches, applying them to data we have scraped from a popular anime database/community website.

The three different approaches we will be exploring are
<ul>
<li>- Content Based Filtering</li>
<li>- Collaborative Filtering</li>
<li>- Hybrid (Combination of the first two approaches)</li>
</ul>


For the sake of comparison and evaluation, there will also be a baseline model that provides recommendations based on the most popular titles that users have not interacted with.

<a id='2'></a>
## 2. Data
We will be using [scraped datasets](https://wenhao7.github.io/data_wrangling/misc/2024/04/12/mal_scrape_part1.html) that we have obtained previously. There are 3 separate datasets that can provide us the data we need to create the recommendation systems.
<ul>
<li>- cleaned_anime_info.csv</li>
<li>- cleaned_anime_reviews.csv</li>
<li>- cleaned_user_ratings.csv</li>
</ul>



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from ast import literal_eval
from collections import defaultdict
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse.linalg import svds
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from multiprocessing.pool import ThreadPool as Pool
from datetime import datetime
import random

import warnings
warnings.filterwarnings('ignore')
```

#### Content Data From Anime Titles


```python
df_info = pd.read_csv('cleaned_anime_info.csv')
df_info.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAL_Id</th>
      <th>Name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Status</th>
      <th>Producers</th>
      <th>Licensors</th>
      <th>Studios</th>
      <th>Source</th>
      <th>Genres</th>
      <th>...</th>
      <th>Score-2</th>
      <th>Score-1</th>
      <th>Synopsis</th>
      <th>Voice_Actors</th>
      <th>Recommended_Ids</th>
      <th>Recommended_Counts</th>
      <th>Aired_Start</th>
      <th>Aired_End</th>
      <th>Premiered_Season</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52991</td>
      <td>Sousou no Frieren</td>
      <td>TV</td>
      <td>28.0</td>
      <td>Finished Airing</td>
      <td>['Aniplex', 'Dentsu', 'Shogakukan-Shueisha Pro...</td>
      <td>['None found', 'add some']</td>
      <td>['Madhouse']</td>
      <td>Manga</td>
      <td>['Adventure', 'Drama', 'Fantasy', 'Shounen']</td>
      <td>...</td>
      <td>402</td>
      <td>4100</td>
      <td>During their decade-long quest to defeat the D...</td>
      <td>['Tanezaki, Atsumi', 'Ichinose, Kana', 'Kobaya...</td>
      <td>['33352', '41025', '35851', '486', '457', '296...</td>
      <td>['14', '11', '8', '5', '5', '4', '4', '3', '2'...</td>
      <td>2023-09-29</td>
      <td>2024-03-22</td>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5114</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>TV</td>
      <td>64.0</td>
      <td>Finished Airing</td>
      <td>['Aniplex', 'Square Enix', 'Mainichi Broadcast...</td>
      <td>['Funimation', 'Aniplex of America']</td>
      <td>['Bones']</td>
      <td>Manga</td>
      <td>['Action', 'Adventure', 'Drama', 'Fantasy', 'M...</td>
      <td>...</td>
      <td>3460</td>
      <td>50602</td>
      <td>After a horrific alchemy experiment goes wrong...</td>
      <td>['Park, Romi', 'Kugimiya, Rie', 'Miki, Shinich...</td>
      <td>['11061', '16498', '1482', '38000', '9919', '1...</td>
      <td>['74', '44', '21', '17', '16', '14', '14', '9'...</td>
      <td>2009-04-05</td>
      <td>2010-07-04</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9253</td>
      <td>Steins;Gate</td>
      <td>TV</td>
      <td>24.0</td>
      <td>Finished Airing</td>
      <td>['Frontier Works', 'Media Factory', 'Kadokawa ...</td>
      <td>['Funimation']</td>
      <td>['White Fox']</td>
      <td>Visual novel</td>
      <td>['Drama', 'Sci-Fi', 'Suspense', 'Psychological...</td>
      <td>...</td>
      <td>2868</td>
      <td>10054</td>
      <td>Eccentric scientist Rintarou Okabe has a never...</td>
      <td>['Miyano, Mamoru', 'Imai, Asami', 'Hanazawa, K...</td>
      <td>['31043', '31240', '9756', '10620', '2236', '4...</td>
      <td>['132', '130', '48', '26', '24', '19', '19', '...</td>
      <td>2011-04-06</td>
      <td>2011-09-14</td>
      <td>2.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28977</td>
      <td>Gintama°</td>
      <td>TV</td>
      <td>51.0</td>
      <td>Finished Airing</td>
      <td>['TV Tokyo', 'Aniplex', 'Dentsu']</td>
      <td>['Funimation', 'Crunchyroll']</td>
      <td>['Bandai Namco Pictures']</td>
      <td>Manga</td>
      <td>['Action', 'Comedy', 'Sci-Fi', 'Gag Humor', 'H...</td>
      <td>...</td>
      <td>1477</td>
      <td>8616</td>
      <td>Gintoki, Shinpachi, and Kagura return as the f...</td>
      <td>['Sugita, Tomokazu', 'Kugimiya, Rie', 'Sakaguc...</td>
      <td>['9863', '30276', '33255', '37105', '6347', '3...</td>
      <td>['3', '2', '1', '1', '1', '1', '1', '1', '1', ...</td>
      <td>2015-04-08</td>
      <td>2016-03-30</td>
      <td>2.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38524</td>
      <td>Shingeki no Kyojin Season 3 Part 2</td>
      <td>TV</td>
      <td>10.0</td>
      <td>Finished Airing</td>
      <td>['Production I.G', 'Dentsu', 'Mainichi Broadca...</td>
      <td>['Funimation']</td>
      <td>['Wit Studio']</td>
      <td>Manga</td>
      <td>['Action', 'Drama', 'Suspense', 'Gore', 'Milit...</td>
      <td>...</td>
      <td>1308</td>
      <td>12803</td>
      <td>Seeking to restore humanity's diminishing hope...</td>
      <td>['Kamiya, Hiroshi', 'Kaji, Yuuki', 'Ishikawa, ...</td>
      <td>['28623', '37521', '25781', '2904', '36649', '...</td>
      <td>['1', '1', '1', '1', '1', '1', '1', '1', '1', ...</td>
      <td>2019-04-29</td>
      <td>2019-07-01</td>
      <td>2.0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
df_info.columns
```




    Index(['MAL_Id', 'Name', 'Type', 'Episodes', 'Status', 'Producers',
           'Licensors', 'Studios', 'Source', 'Genres', 'Duration', 'Rating',
           'Score', 'Popularity', 'Members', 'Favorites', 'Watching', 'Completed',
           'On-Hold', 'Dropped', 'Plan to Watch', 'Total', 'Score-10', 'Score-9',
           'Score-8', 'Score-7', 'Score-6', 'Score-5', 'Score-4', 'Score-3',
           'Score-2', 'Score-1', 'Synopsis', 'Voice_Actors', 'Recommended_Ids',
           'Recommended_Counts', 'Aired_Start', 'Aired_End', 'Premiered_Season',
           'Rank'],
          dtype='object')



If we want to create a model that will provide recommendations based on the contents of a title, features such as 'Recommended_Counts', 'Aired_Start', 'Aired_End', 'Premiered_Season', 'Rank', and the granular Scores/Interaction features can be excluded when modeling as they do not provide valuable information. 

We can retain the 'Score' and 'Popularity' metrics to serve as aggregates of the contents of each title, treating them as attributes of their titles as they do not provide any granular information on user preferences.

#### Hybrid Data From User Reviews


```python
df_review = pd.read_csv('cleaned_anime_reviews.csv')
df_review.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review_id</th>
      <th>MAL_Id</th>
      <th>Review</th>
      <th>Tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>52991</td>
      <td>With lives so short, why do we even bother? To...</td>
      <td>Recommended</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>52991</td>
      <td>With lives so short, why do we even bother? To...</td>
      <td>Preliminary</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>52991</td>
      <td>Frieren is the most overrated anime of this de...</td>
      <td>Not-Recommended</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>52991</td>
      <td>Frieren is the most overrated anime of this de...</td>
      <td>Funny</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>52991</td>
      <td>Frieren is the most overrated anime of this de...</td>
      <td>Preliminary</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_review.Tags.value_counts()
```




    Tags
    Recommended        48344
    Mixed-Feelings     15160
    Not-Recommended    14413
    Preliminary        13187
    Funny                846
    Well-written         250
    Informative          130
    Creative               2
    Name: count, dtype: int64




```python
# Only retain Recommended, Mixed-Feelings, Not-Recommended, for collaborative data
# Entire Review data will be vectorized to provide content and collaborative data
```

An approach to using the review dataset would be to process the text information from each review to extract additional content information provided for the titles by actual users, and the tags associated with each review can be processed to get user sentiments.

However, we will be skipping the reviews dataset in this notebook to focus more on the creation of the recommendation systems with the other two datasets. Further exploration of including these review data will be conducted in the future.

#### Collaborative Data From User Ratings


```python
df_ratings = pd.read_csv('cleaned_user_ratings.csv')
df_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Username</th>
      <th>User_Id</th>
      <th>Anime_Id</th>
      <th>Anime_Title</th>
      <th>Rating_Status</th>
      <th>Rating_Score</th>
      <th>Num_Epi_Watched</th>
      <th>Is_Rewatching</th>
      <th>Updated</th>
      <th>Start_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>flerbz</td>
      <td>0</td>
      <td>30654</td>
      <td>Ansatsu Kyoushitsu 2nd Season</td>
      <td>watching</td>
      <td>0</td>
      <td>24</td>
      <td>False</td>
      <td>2022-02-26 22:15:01+00:00</td>
      <td>2022-01-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>flerbz</td>
      <td>0</td>
      <td>22789</td>
      <td>Barakamon</td>
      <td>dropped</td>
      <td>0</td>
      <td>2</td>
      <td>False</td>
      <td>2023-01-28 19:03:33+00:00</td>
      <td>2022-04-06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>flerbz</td>
      <td>0</td>
      <td>31964</td>
      <td>Boku no Hero Academia</td>
      <td>completed</td>
      <td>0</td>
      <td>13</td>
      <td>False</td>
      <td>2024-03-31 02:10:32+00:00</td>
      <td>2024-03-30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>flerbz</td>
      <td>0</td>
      <td>33486</td>
      <td>Boku no Hero Academia 2nd Season</td>
      <td>completed</td>
      <td>0</td>
      <td>25</td>
      <td>False</td>
      <td>2024-03-31 22:32:02+00:00</td>
      <td>2024-03-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>flerbz</td>
      <td>0</td>
      <td>36456</td>
      <td>Boku no Hero Academia 3rd Season</td>
      <td>watching</td>
      <td>0</td>
      <td>24</td>
      <td>False</td>
      <td>2024-04-03 02:08:56+00:00</td>
      <td>2024-03-31</td>
    </tr>
  </tbody>
</table>
</div>



Within this dataset our main feature would be 'Rating_Score' and their corresponding 'User_Id' and 'Anime_Id', so that we can map out each user's preferences and use that information to find other users with similar preferences to obtain recommendations from. 

'Rating_Status' may also be a useful feature that will tell us how the user has interacted with a particular title. "Planning to watch" a title suggests that the user knows of and is already interested in the title, while "Completed" will tells us that the user likes the title enough to finish it, making "Completed" > "Planning to watch" in terms of user interaction.

Next, we would also be interested in removing users that have too few entries in their list as they will just increase computational load without providing the same level of information for our model.


```python
tmp = [(df_ratings.value_counts('User_Id')>=i).sum() for i in range(5,305,5)]
tmp = pd.DataFrame({"Cutoff":list(range(5,305,5)), "Users":tmp})
sns.lineplot(tmp, x='Cutoff', y='Users')
```




    <Axes: xlabel='Cutoff', ylabel='Users'>




    
![png](/images/mal_rec_exploration/output_12_1.png)
    


The number of users decreases linearly with increasing cutoff points (number of titles in their list). Since there is not an obvious cutoff that we can select from this, we shall arbitrarily set it to 20 interactions removing all users that have less than 20 titles in their list.


```python
print(f'Number of unique users : {df_ratings.User_Id.nunique()}')
print(f'Number of user interactions : {df_ratings.shape[0]}')
```

    Number of unique users : 17513
    Number of user interactions : 5452192
    


```python
tmp = (df_ratings.value_counts('User_Id') >= 20).reset_index()
tmp = tmp[tmp['count']==True]
df_ratings = df_ratings[df_ratings.User_Id.isin(tmp.User_Id)]
```


```python
print(f'After removing users with less than 20 interactions')
print(f'Number of unique users : {df_ratings.User_Id.nunique()}')
print(f'Number of user interactions : {df_ratings.shape[0]}')
```

    After removing users with less than 20 interactions
    Number of unique users : 16744
    Number of user interactions : 5445702
    

We see that we have 16744 users left, a decrease of aout 4.4% from the original 17513 users.

<a id='3'></a>
## 3. Baseline Model & Evaluation
As a baseline we will simply recommend users the most popular titles (based on popularity metric) that is not on their ratings list.


```python
df_info[['MAL_Id','Name','Score','Popularity','Rank']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAL_Id</th>
      <th>Name</th>
      <th>Score</th>
      <th>Popularity</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52991</td>
      <td>Sousou no Frieren</td>
      <td>9.276142</td>
      <td>301</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5114</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>8.941080</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9253</td>
      <td>Steins;Gate</td>
      <td>8.962588</td>
      <td>13</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28977</td>
      <td>Gintama°</td>
      <td>8.726812</td>
      <td>341</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38524</td>
      <td>Shingeki no Kyojin Season 3 Part 2</td>
      <td>9.019487</td>
      <td>21</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
def mask_user_ratings(user_ratings_df, random_state=42):
    # Sample half of rated titles as input
    input_df = user_ratings_df[user_ratings_df['Rating_Score']>0].sample(frac=0.5, random_state=random_state)
    val_df = user_ratings_df.drop(input_df.index)
    return input_df, val_df
```

The above function splits the user ratings dataset into an input and validation splits by sampling a user's rated titles into the input split and placing the remaining titles into the validation split.

This approach will allow us to provide a subset of the ground truth (user's ratings) into our recommendation system as input, and evaluate the recommendations against the remaining subset of the ground truth.


```python
class PopularityRec:
    def __init__(self, df, anime_info_df=None):
        self.popularity = df
        self.anime_info = anime_info_df
        
    def predict(self, user_ratings_df, topn=10, left_on='MAL_Id', right_on='Anime_Id'):
        rec_df = self.popularity.sort_values('Popularity', ascending=True)
        rec_df = rec_df.merge(user_ratings_df, how='left', left_on=left_on, right_on=right_on)
        return rec_df.loc[rec_df[right_on].isna()][self.popularity.columns][:topn]
```

The above code creates a Popularity Recommendation System object that will help create predictions based on the most popular titles that the input user has not interacted with, and will be the baseline model in this notebook.

### Evaluation
Two metrics will be used here

1. Mean Reciprocal Rank

    $$MRR = \dfrac{1}{N}\sum_{n=1}^N \dfrac{1}{rank_i}$$

Where N is the total number of users, n is the nth user, i is the position of the first relevant item found within the recommendations

2. Normalized Discounted Cumulative Gain

    $$NDCG@K = \dfrac{DCG@K}{IdealDCG@K}$$
    $$DCG@K = \sum_{k=1}^K \dfrac{rel_k}{log_2 (k+1)}$$

Where K is the total number of top recommendations we are evaluating, k is the kth highest predicted recommendation, rel_k is the relevance score of the recommendation at position k

IdealDCG@K is calculated by sorted the recommendations@K from order of highest relevance to lowest relevance before calculating the DGC@K. This will return the maximum achieveable DCG for the same set of ranked recommendations@K. 

Essentially, the evaluation process in this notebook would be:
<ul>
<li>- Each user will have part of their ratings list randomly sampled to use as input data</li>
<li>- Recommendation system takes in input data and returns a set of ranked recommendations</li>
<li>- Remaining part of the user's rating list will be used as validation data</li>
<li>- MRR/NDCG will be calcualted from this validation data recommendations and the masked portion of the user's rating list</li>
</ul>


```python
class ModelEvaluator:
    def evaluate_mrr(self, ranked_rec_df, user_input_df, user_val_df, weight=1, topn=10, left_on='MAL_Id', right_on='Anime_Id'):
        scoring_df = ranked_rec_df.merge(user_val_df, how='left', left_on=left_on, right_on=right_on)
        scoring_df = scoring_df.loc[~scoring_df[right_on].isna()][:topn]
        matched_idx = list(scoring_df[scoring_df[right_on].isin(user_val_df[right_on])].index)
        if not matched_idx:
            return 0
        return (1 * weight) / (matched_idx[0] + 1)
    
    def evaluate_ndcg(self, ranked_rec_df, user_input_df, user_val_df, weight=1, topn=10, left_on='MAL_Id', right_on='Anime_Id'):
        scoring_df = ranked_rec_df.merge(user_val_df, how='left', left_on=left_on, right_on=right_on)
        scoring_df = scoring_df.iloc[:topn]
        # Calculate relevance score based on how well the user interaction went
        for i in range(len(scoring_df)):
            scoring_df['rel'] = 0.0
            scoring_df.loc[scoring_df.Rating_Score == 0, 'rel'] = 0.5
            scoring_df.loc[scoring_df.Rating_Score > 0, 'rel'] = 1
            scoring_df.loc[scoring_df.Rating_Score > 5, 'rel'] = 2
            scoring_df.loc[scoring_df.Rating_Score > 8, 'rel'] = 3
            
        cg, icg = list(scoring_df['rel']) , sorted(scoring_df['rel'], reverse=True)            
        if not cg or max(cg) == 0:
            return 0
        icg = sorted(cg, reverse=True)
        cg = list(np.array(cg) / np.array([math.log(i+1, 2) for i in range(1,len(cg) + 1)]))
        icg = list(np.array(icg) / np.array([math.log(i+1, 2) for i in range(1,len(icg) + 1)]))
        ndcg = sum(cg) / sum(icg)
        return ndcg
    
```


```python
test_input_df, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==47], random_state=1) 
```


```python
tester_rec = PopularityRec(df_info, df_info)
pred_df = tester_rec.predict(test_input_df, topn=10)
pred_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAL_Id</th>
      <th>Name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Status</th>
      <th>Producers</th>
      <th>Licensors</th>
      <th>Studios</th>
      <th>Source</th>
      <th>Genres</th>
      <th>...</th>
      <th>Score-2</th>
      <th>Score-1</th>
      <th>Synopsis</th>
      <th>Voice_Actors</th>
      <th>Recommended_Ids</th>
      <th>Recommended_Counts</th>
      <th>Aired_Start</th>
      <th>Aired_End</th>
      <th>Premiered_Season</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16498</td>
      <td>Shingeki no Kyojin</td>
      <td>TV</td>
      <td>25.0</td>
      <td>Finished Airing</td>
      <td>['Production I.G', 'Dentsu', 'Mainichi Broadca...</td>
      <td>['Funimation']</td>
      <td>['Wit Studio']</td>
      <td>Manga</td>
      <td>['Action', 'Award Winning', 'Drama', 'Suspense...</td>
      <td>...</td>
      <td>3828</td>
      <td>9049</td>
      <td>Centuries ago, mankind was slaughtered to near...</td>
      <td>['Kaji, Yuuki', 'Ishikawa, Yui', 'Inoue, Marin...</td>
      <td>['28623', '37779', '26243', '20787', '5114', '...</td>
      <td>['111', '49', '49', '45', '44', '42', '36', '3...</td>
      <td>2013-04-07</td>
      <td>2013-09-29</td>
      <td>2.0</td>
      <td>109</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1535</td>
      <td>Death Note</td>
      <td>TV</td>
      <td>37.0</td>
      <td>Finished Airing</td>
      <td>['VAP', 'Nippon Television Network', 'Shueisha...</td>
      <td>['VIZ Media']</td>
      <td>['Madhouse']</td>
      <td>Manga</td>
      <td>['Supernatural', 'Suspense', 'Psychological', ...</td>
      <td>...</td>
      <td>3238</td>
      <td>5382</td>
      <td>Brutal murders, petty thefts, and senseless vi...</td>
      <td>['Yamaguchi, Kappei', 'Miyano, Mamoru', 'Nakam...</td>
      <td>['1575', '19', '23283', '10620', '13601', '290...</td>
      <td>['633', '113', '95', '74', '67', '52', '50', '...</td>
      <td>2006-10-04</td>
      <td>2007-06-27</td>
      <td>4.0</td>
      <td>79</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5114</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>TV</td>
      <td>64.0</td>
      <td>Finished Airing</td>
      <td>['Aniplex', 'Square Enix', 'Mainichi Broadcast...</td>
      <td>['Funimation', 'Aniplex of America']</td>
      <td>['Bones']</td>
      <td>Manga</td>
      <td>['Action', 'Adventure', 'Drama', 'Fantasy', 'M...</td>
      <td>...</td>
      <td>3460</td>
      <td>50602</td>
      <td>After a horrific alchemy experiment goes wrong...</td>
      <td>['Park, Romi', 'Kugimiya, Rie', 'Miki, Shinich...</td>
      <td>['11061', '16498', '1482', '38000', '9919', '1...</td>
      <td>['74', '44', '21', '17', '16', '14', '14', '9'...</td>
      <td>2009-04-05</td>
      <td>2010-07-04</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30276</td>
      <td>One Punch Man</td>
      <td>TV</td>
      <td>12.0</td>
      <td>Finished Airing</td>
      <td>['TV Tokyo', 'Bandai Visual', 'Lantis', 'Asats...</td>
      <td>['VIZ Media']</td>
      <td>['Madhouse']</td>
      <td>Web manga</td>
      <td>['Action', 'Comedy', 'Adult Cast', 'Parody', '...</td>
      <td>...</td>
      <td>2027</td>
      <td>3701</td>
      <td>The seemingly unimpressive Saitama has a rathe...</td>
      <td>['Furukawa, Makoto', 'Ishikawa, Kaito', 'Yuuki...</td>
      <td>['32182', '31964', '33255', '29803', '918', '5...</td>
      <td>['163', '94', '26', '21', '16', '16', '11', '1...</td>
      <td>2015-10-05</td>
      <td>2015-12-21</td>
      <td>4.0</td>
      <td>129</td>
    </tr>
    <tr>
      <th>5</th>
      <td>38000</td>
      <td>Kimetsu no Yaiba</td>
      <td>TV</td>
      <td>26.0</td>
      <td>Finished Airing</td>
      <td>['Aniplex', 'Shueisha']</td>
      <td>['Aniplex of America']</td>
      <td>['ufotable']</td>
      <td>Manga</td>
      <td>['Action', 'Award Winning', 'Fantasy', 'Histor...</td>
      <td>...</td>
      <td>2354</td>
      <td>6186</td>
      <td>Ever since the death of his father, the burden...</td>
      <td>['Hanae, Natsuki', 'Shimono, Hiro', 'Kitou, Ak...</td>
      <td>['40748', '37520', '16498', '269', '5114', '31...</td>
      <td>['70', '42', '20', '20', '17', '15', '12', '11...</td>
      <td>2019-04-06</td>
      <td>2019-09-28</td>
      <td>2.0</td>
      <td>143</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>



As a sanity check we have made some recommendations using User Id 47's ratings list as shown above.


```python
tester_eval = ModelEvaluator()
print('MRR : ', tester_eval.evaluate_mrr(pred_df, test_input_df, test_val_df, topn=10))
print('NDCG : ', tester_eval.evaluate_ndcg(pred_df, test_input_df, test_val_df, topn=10))
```

    MRR :  0.2
    NDCG :  0.430624116386567
    

Evaluating the predictions also reveals the above scores.

Next we shall make and evaluate recommendations for all the users in our dataset.


```python
#calculate baseline performance
count = 0
total_mrr = 0
total_ndcg = 0
mrr_base, ndcg_base = [], []
for i in df_ratings.User_Id.unique():
    count += 1
    test_input_df, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==i], random_state=1) 
    pred_df = tester_rec.predict(test_input_df, topn=10)
    mrr = tester_eval.evaluate_mrr(pred_df, test_input_df, test_val_df, topn=10)
    ndcg = tester_eval.evaluate_ndcg(pred_df, test_input_df, test_val_df, topn=10)
    total_mrr += mrr
    total_ndcg += ndcg
    mrr_base.append(mrr)
    ndcg_base.append(ndcg)
```


```python
def running_avg(scores):
    avgs = np.cumsum(scores)/np.array(range(1, len(scores) + 1))
    return avgs
```


```python
print(f'Baseline MRR: {running_avg(mrr_base)[-1]}')
print(f'Baseline NDCG: {running_avg(ndcg_base)[-1]}')
```

    Baseline MRR: 0.657765657823893
    Baseline NDCG: 0.6612426800967172
    

Above we see our baseline performance that we want to beat.

<a id='4'></a>
## 4. Content Based Filtering
In this section we will explore content based filtering, where only information about the titles (i.e. descriptions, content attributes) will be used to recommend similar items to users based on their stated preferences.

As stated earlier in this notebook, we will be treating "Score" and "Popularity" features as content attributes tied to each title as they do not provide any granular user preferences information.

The underlying assumption here is that users who have interacted with certain of titles will likely enjoy other similar titles. A limitation of this is that recommendations may not be as diverse, where for example a user who has interacted with mostly Action titles will mainly be recommended similar Action titles with little deviation into other types of titles.

Titles similarities will be calculated using cosine similarity, where two similar titles should be pointing in the same direction within an inner product space.

First we will start with some data preprocessing.


```python
df_content = df_info.copy().drop(['Aired_Start','Aired_End','Premiered_Season','Rank','Recommended_Ids','Recommended_Counts','Score-10', 'Score-9',
       'Score-8', 'Score-7', 'Score-6', 'Score-5', 'Score-4', 'Score-3',
       'Score-2', 'Score-1','Total','Watching','Completed','On-Hold','Dropped','Plan to Watch','Status','Source'], axis=1)
df_content.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAL_Id</th>
      <th>Name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Producers</th>
      <th>Licensors</th>
      <th>Studios</th>
      <th>Genres</th>
      <th>Duration</th>
      <th>Rating</th>
      <th>Score</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Synopsis</th>
      <th>Voice_Actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52991</td>
      <td>Sousou no Frieren</td>
      <td>TV</td>
      <td>28.0</td>
      <td>['Aniplex', 'Dentsu', 'Shogakukan-Shueisha Pro...</td>
      <td>['None found', 'add some']</td>
      <td>['Madhouse']</td>
      <td>['Adventure', 'Drama', 'Fantasy', 'Shounen']</td>
      <td>24 min. per ep.</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>9.276142</td>
      <td>301</td>
      <td>670859</td>
      <td>35435</td>
      <td>During their decade-long quest to defeat the D...</td>
      <td>['Tanezaki, Atsumi', 'Ichinose, Kana', 'Kobaya...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5114</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>TV</td>
      <td>64.0</td>
      <td>['Aniplex', 'Square Enix', 'Mainichi Broadcast...</td>
      <td>['Funimation', 'Aniplex of America']</td>
      <td>['Bones']</td>
      <td>['Action', 'Adventure', 'Drama', 'Fantasy', 'M...</td>
      <td>24 min. per ep.</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>8.941080</td>
      <td>3</td>
      <td>3331144</td>
      <td>225215</td>
      <td>After a horrific alchemy experiment goes wrong...</td>
      <td>['Park, Romi', 'Kugimiya, Rie', 'Miki, Shinich...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9253</td>
      <td>Steins;Gate</td>
      <td>TV</td>
      <td>24.0</td>
      <td>['Frontier Works', 'Media Factory', 'Kadokawa ...</td>
      <td>['Funimation']</td>
      <td>['White Fox']</td>
      <td>['Drama', 'Sci-Fi', 'Suspense', 'Psychological...</td>
      <td>24 min. per ep.</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.962588</td>
      <td>13</td>
      <td>2553356</td>
      <td>189031</td>
      <td>Eccentric scientist Rintarou Okabe has a never...</td>
      <td>['Miyano, Mamoru', 'Imai, Asami', 'Hanazawa, K...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28977</td>
      <td>Gintama°</td>
      <td>TV</td>
      <td>51.0</td>
      <td>['TV Tokyo', 'Aniplex', 'Dentsu']</td>
      <td>['Funimation', 'Crunchyroll']</td>
      <td>['Bandai Namco Pictures']</td>
      <td>['Action', 'Comedy', 'Sci-Fi', 'Gag Humor', 'H...</td>
      <td>24 min. per ep.</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.726812</td>
      <td>341</td>
      <td>628071</td>
      <td>16610</td>
      <td>Gintoki, Shinpachi, and Kagura return as the f...</td>
      <td>['Sugita, Tomokazu', 'Kugimiya, Rie', 'Sakaguc...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38524</td>
      <td>Shingeki no Kyojin Season 3 Part 2</td>
      <td>TV</td>
      <td>10.0</td>
      <td>['Production I.G', 'Dentsu', 'Mainichi Broadca...</td>
      <td>['Funimation']</td>
      <td>['Wit Studio']</td>
      <td>['Action', 'Drama', 'Suspense', 'Gore', 'Milit...</td>
      <td>23 min. per ep.</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>9.019487</td>
      <td>21</td>
      <td>2262916</td>
      <td>58383</td>
      <td>Seeking to restore humanity's diminishing hope...</td>
      <td>['Kamiya, Hiroshi', 'Kaji, Yuuki', 'Ishikawa, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Convert duration column to number of minutes
def convert_duration(duration):
    duration = duration.split(' ')
    duration_mins = 0
    curr_min = 1/60
    for char in duration[::-1]:
        if 'min' in char:
            curr_min = 1
        elif 'hr' in char:
            curr_min = 60
        elif char.isnumeric():
            duration_mins += int(char) * curr_min
    return duration_mins
```


```python
df_content.Duration = df_content.Duration.apply(convert_duration)
df_content.Duration.head()
```




    0    24.0
    1    24.0
    2    24.0
    3    24.0
    4    23.0
    Name: Duration, dtype: float64




```python
# Onehotencode Genre
genres = df_content['Genres'].apply(literal_eval).explode()
genres = 'genre_' + genres
genres = genres.fillna('genre_na')
df_content = df_content.drop('Genres', axis = 1).join(pd.crosstab(genres.index, genres))
df_content.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAL_Id</th>
      <th>Name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Producers</th>
      <th>Licensors</th>
      <th>Studios</th>
      <th>Duration</th>
      <th>Rating</th>
      <th>Score</th>
      <th>...</th>
      <th>genre_Supernatural</th>
      <th>genre_Survival</th>
      <th>genre_Suspense</th>
      <th>genre_Team Sports</th>
      <th>genre_Time Travel</th>
      <th>genre_Vampire</th>
      <th>genre_Video Game</th>
      <th>genre_Visual Arts</th>
      <th>genre_Workplace</th>
      <th>genre_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52991</td>
      <td>Sousou no Frieren</td>
      <td>TV</td>
      <td>28.0</td>
      <td>['Aniplex', 'Dentsu', 'Shogakukan-Shueisha Pro...</td>
      <td>['None found', 'add some']</td>
      <td>['Madhouse']</td>
      <td>24.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>9.276142</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5114</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>TV</td>
      <td>64.0</td>
      <td>['Aniplex', 'Square Enix', 'Mainichi Broadcast...</td>
      <td>['Funimation', 'Aniplex of America']</td>
      <td>['Bones']</td>
      <td>24.0</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>8.941080</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9253</td>
      <td>Steins;Gate</td>
      <td>TV</td>
      <td>24.0</td>
      <td>['Frontier Works', 'Media Factory', 'Kadokawa ...</td>
      <td>['Funimation']</td>
      <td>['White Fox']</td>
      <td>24.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.962588</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28977</td>
      <td>Gintama°</td>
      <td>TV</td>
      <td>51.0</td>
      <td>['TV Tokyo', 'Aniplex', 'Dentsu']</td>
      <td>['Funimation', 'Crunchyroll']</td>
      <td>['Bandai Namco Pictures']</td>
      <td>24.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.726812</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38524</td>
      <td>Shingeki no Kyojin Season 3 Part 2</td>
      <td>TV</td>
      <td>10.0</td>
      <td>['Production I.G', 'Dentsu', 'Mainichi Broadca...</td>
      <td>['Funimation']</td>
      <td>['Wit Studio']</td>
      <td>23.0</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>9.019487</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 90 columns</p>
</div>




```python
# Labelencode Type, Rating
cols = ['Type','Rating']
for col in cols:
    le = LabelEncoder()
    df_content[col] = le.fit_transform(df_content[col])
df_content[cols].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Above we have encoded the relevant categorical features found in the dataset, next we will need to vectorize the remaining features.


```python
# Count Vectorize Name, Producers, Licensors, Studios, Voice_Actors, 
cols = ['Name','Producers','Licensors','Studios','Voice_Actors']
sparse_total=[]
for col in cols:
    df_content[col].apply(lambda x: '' if pd.isna(x) else x.strip('[]'))
    vec = CountVectorizer()
    tmp = df_content[col]
    sparse_tmp = vec.fit_transform(tmp)
    if isinstance(sparse_total,list):
        sparse_total=sparse_tmp
    else:
        sparse_total = hstack((sparse_total, sparse_tmp))

sparse_total    
```




    <13300x18768 sparse matrix of type '<class 'numpy.int64'>'
    	with 332269 stored elements in Compressed Sparse Row format>




```python
# TFIDF Vectorize Synopsis to place emphasis on words with less occurences
sw = stopwords.words('english')
tfidf_vec = TfidfVectorizer(analyzer='word',
                            ngram_range=(1,2),
                            max_df=0.5,
                            min_df=0.001,
                            stop_words=sw)
```


```python
sparse_tfidf = tfidf_vec.fit_transform(df_content['Synopsis'])
sparse_tfidf
```




    <13300x6754 sparse matrix of type '<class 'numpy.float64'>'
    	with 488956 stored elements in Compressed Sparse Row format>



For this approach we will not be combining the various arrays into a single dataframe as input. Instead we will leave them in three separate arrays
<ul>
<li>- Dense dataframe containing numerical and categorical columns</li>
<li>- Sparse array containing count vectorized features</li>
<li>- Sparse array containing tfdidf vectorized synopsis</li>
</ul>

to decrease computational costs when calculating similarities between titles in the array. Recommendations made from each of the three arrays will contribute to a final list of recommendations.


```python
df_dense = df_content.drop(['Name','Producers','Licensors','Studios','Voice_Actors','Synopsis'], axis=1)
df_dense.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAL_Id</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Duration</th>
      <th>Rating</th>
      <th>Score</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>genre_Action</th>
      <th>...</th>
      <th>genre_Supernatural</th>
      <th>genre_Survival</th>
      <th>genre_Suspense</th>
      <th>genre_Team Sports</th>
      <th>genre_Time Travel</th>
      <th>genre_Vampire</th>
      <th>genre_Video Game</th>
      <th>genre_Visual Arts</th>
      <th>genre_Workplace</th>
      <th>genre_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52991</td>
      <td>4</td>
      <td>28.0</td>
      <td>24.0</td>
      <td>2</td>
      <td>9.276142</td>
      <td>301</td>
      <td>670859</td>
      <td>35435</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5114</td>
      <td>4</td>
      <td>64.0</td>
      <td>24.0</td>
      <td>3</td>
      <td>8.941080</td>
      <td>3</td>
      <td>3331144</td>
      <td>225215</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9253</td>
      <td>4</td>
      <td>24.0</td>
      <td>24.0</td>
      <td>2</td>
      <td>8.962588</td>
      <td>13</td>
      <td>2553356</td>
      <td>189031</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28977</td>
      <td>4</td>
      <td>51.0</td>
      <td>24.0</td>
      <td>2</td>
      <td>8.726812</td>
      <td>341</td>
      <td>628071</td>
      <td>16610</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38524</td>
      <td>4</td>
      <td>10.0</td>
      <td>23.0</td>
      <td>3</td>
      <td>9.019487</td>
      <td>21</td>
      <td>2262916</td>
      <td>58383</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>




```python
print("Missing Episodes: ", df_dense.Episodes.isna().sum())
df_dense.Episodes = df_dense.Episodes.fillna(0)
print("Missing Episodes after fillna: ", df_dense.Episodes.isna().sum())
```

    Missing Episodes:  55
    Missing Episodes after fillna:  0
    


```python
scale_cols = ['Score','Members','Favorites','Episodes']
ss = StandardScaler()
df_dense[scale_cols] = ss.fit_transform(df_dense[scale_cols])
df_dense.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAL_Id</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Duration</th>
      <th>Rating</th>
      <th>Score</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>genre_Action</th>
      <th>...</th>
      <th>genre_Supernatural</th>
      <th>genre_Survival</th>
      <th>genre_Suspense</th>
      <th>genre_Team Sports</th>
      <th>genre_Time Travel</th>
      <th>genre_Vampire</th>
      <th>genre_Video Game</th>
      <th>genre_Visual Arts</th>
      <th>genre_Workplace</th>
      <th>genre_na</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52991</td>
      <td>4</td>
      <td>0.271027</td>
      <td>24.0</td>
      <td>2</td>
      <td>2.745298</td>
      <td>301</td>
      <td>2.705886</td>
      <td>5.556849</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5114</td>
      <td>4</td>
      <td>0.952556</td>
      <td>24.0</td>
      <td>3</td>
      <td>2.419119</td>
      <td>3</td>
      <td>14.743216</td>
      <td>36.053453</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9253</td>
      <td>4</td>
      <td>0.195302</td>
      <td>24.0</td>
      <td>2</td>
      <td>2.440056</td>
      <td>13</td>
      <td>11.223860</td>
      <td>30.238883</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28977</td>
      <td>4</td>
      <td>0.706449</td>
      <td>24.0</td>
      <td>2</td>
      <td>2.210530</td>
      <td>341</td>
      <td>2.512278</td>
      <td>2.531775</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38524</td>
      <td>4</td>
      <td>-0.069737</td>
      <td>23.0</td>
      <td>3</td>
      <td>2.495447</td>
      <td>21</td>
      <td>9.909669</td>
      <td>9.244467</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 84 columns</p>
</div>




```python
class ContentBasedRecommender:
    def __init__(self, df_content):
        self.df_content = df_content
        self.df_dense, self.sparse_vec, self.sparse_tfidf = self.process_df(self.df_content)
        self.ref_weights = [1/math.log(len(self.df_content)-i+1, 10) + 1 for i in range(len(self.df_content))]
        
    def process_df(self, df_content):
        genres=df_content['Genres'].apply(literal_eval).explode()
        genres = 'genre_' + genres
        genres = genres.fillna('genre_na')
        df_content = df_content.drop('Genres', axis=1).join(pd.crosstab(genres.index, genres))
        
        #labelencode
        for col in ['Type','Rating']:
            le=LabelEncoder()
            df_content[col] = le.fit_transform(df_content[col])
        #Vectorize
        sparse_vec=[]
        for col in ['Name','Producers','Licensors','Studios','Voice_Actors']:
            df_content[col].apply(lambda x: '' if pd.isna(x) else x.strip('[]'))
            vec = CountVectorizer()
            tmp = df_content[col]
            sparse_tmp = vec.fit_transform(tmp)
            if isinstance(sparse_vec,list):
                sparse_vec = sparse_tmp
            else:
                sparse_vec = hstack((sparse_vec, sparse_tmp))
        
        tfidf_vec = TfidfVectorizer(analyzer='word',
                            ngram_range=(1,2),
                            max_df=0.5,
                            min_df=0.001,
                            stop_words=sw)
        sparse_tfidf = tfidf_vec.fit_transform(df_content['Synopsis'])
        
        df_dense = df_content.drop(['Name','Producers','Licensors','Studios','Voice_Actors','Synopsis'], axis=1)
        df_dense.Episodes = df_dense.Episodes.fillna(0)
        scale_cols = ['Score','Members','Favorites','Episodes']
        ss = StandardScaler()
        df_dense[scale_cols] = ss.fit_transform(df_dense[scale_cols])
        
        return df_dense, sparse_vec, sparse_tfidf
    
        self.df_dense = df_dense
        self.sparse_vec = sparse_vec
        self.sparse_tfidf = sparse_tfidf
    
    def get_entry(self, MAL_Id):
        title_dense = self.df_dense[self.df_dense['MAL_Id'] == MAL_Id]
        idx = title_dense.index[0]
        title_vec = self.sparse_vec[idx]
        title_tfidf = self.sparse_tfidf[idx]
        return title_dense, title_vec, title_tfidf
    
    def calc_sim(self, MAL_Id):
        try:
            title_dense, title_vec, title_tfidf = self.get_entry(MAL_Id)
        except:
            return None
        sim_dense = cosine_similarity(title_dense, self.df_dense)
        sim_vec = cosine_similarity(title_vec, self.sparse_vec)
        sim_tfidf = cosine_similarity(title_tfidf, self.sparse_tfidf)
        total = (sim_dense + sim_vec + sim_tfidf).argsort().flatten()
        return total
    
    
    def predict_weights(self, user_list):
        weights_df = pd.DataFrame({'Preds': self.df_content.MAL_Id, 'Weights':0})
        for MAL_Id in user_list:
            recs = self.calc_sim(MAL_Id)
            if recs is None:
                continue
            idx_recs = list(recs)
            weights_zip = list(zip(idx_recs, self.ref_weights))
            weights_zip = sorted(weights_zip)
            weights_zip = list(zip(*weights_zip))
            weights_df['Weights'] += weights_zip[1]
        weights_df['Weights'] = (weights_df['Weights'] - weights_df['Weights'].min()) / (weights_df['Weights'].max() - weights_df['Weights'].min())
        return weights_df
    
    def par_weights(self, user_list):
        weights_df = pd.DataFrame({'Preds': self.df_content.MAL_Id, 'Weights':0})
        recs_list=[]
        with Pool() as pool:
            for recs in pool.imap(self.calc_sim, user_list):
                if recs is None:
                    continue
                recs_list.append(recs)
        for recs in recs_list:
            idx_recs = list(recs)
            weights_zip = list(zip(idx_recs, self.ref_weights))
            weights_zip = sorted(weights_zip)
            weights_zip = list(zip(*weights_zip))
            weights_df['Weights'] += weights_zip[1]
        weights_df['Weights'] = (weights_df['Weights'] - weights_df['Weights'].min()) / (weights_df['Weights'].max() - weights_df['Weights'].min())
        return weights_df
    
    def par_predict(self, user_df, topn=10):
        user_list = list(user_df['Anime_Id'])
        weights_df = self.par_weights(user_list)
        res = weights_df.merge(self.df_content, how='left', left_on='Preds', right_on='MAL_Id')
        res = res.sort_values('Weights', ascending=False).loc[~res['MAL_Id'].isin(user_list)][:topn]
        return res
    
    def predict(self, user_df, topn=10):
        user_list = list(user_df['Anime_Id'])
        weights_df = self.predict_weights(user_list)
        res = weights_df.merge(self.df_content, how='left', left_on='Preds', right_on='MAL_Id')
        res = res.sort_values('Weights', ascending=False).loc[~res['MAL_Id'].isin(user_list)][:topn]
        return res        
```

The above code creates a Content Based Recommendation System object that will be able to process input datasets and make recommendations. During experimentation, calculation of cosine similarity was expensive and taking too long, hence within the class I have written functions that will do the calculations in parallel to speed things up.


```python
df_content = df_info.copy().drop(['Aired_Start','Aired_End','Premiered_Season','Rank','Recommended_Ids','Recommended_Counts','Score-10', 'Score-9',
       'Score-8', 'Score-7', 'Score-6', 'Score-5', 'Score-4', 'Score-3',
       'Score-2', 'Score-1','Total','Watching','Completed','On-Hold','Dropped','Plan to Watch','Status','Source'], axis=1)
df_content.Duration = df_content.Duration.apply(convert_duration)
df_content.Duration.head()
df_content.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MAL_Id</th>
      <th>Name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Producers</th>
      <th>Licensors</th>
      <th>Studios</th>
      <th>Genres</th>
      <th>Duration</th>
      <th>Rating</th>
      <th>Score</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Synopsis</th>
      <th>Voice_Actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>52991</td>
      <td>Sousou no Frieren</td>
      <td>TV</td>
      <td>28.0</td>
      <td>['Aniplex', 'Dentsu', 'Shogakukan-Shueisha Pro...</td>
      <td>['None found', 'add some']</td>
      <td>['Madhouse']</td>
      <td>['Adventure', 'Drama', 'Fantasy', 'Shounen']</td>
      <td>24.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>9.276142</td>
      <td>301</td>
      <td>670859</td>
      <td>35435</td>
      <td>During their decade-long quest to defeat the D...</td>
      <td>['Tanezaki, Atsumi', 'Ichinose, Kana', 'Kobaya...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5114</td>
      <td>Fullmetal Alchemist: Brotherhood</td>
      <td>TV</td>
      <td>64.0</td>
      <td>['Aniplex', 'Square Enix', 'Mainichi Broadcast...</td>
      <td>['Funimation', 'Aniplex of America']</td>
      <td>['Bones']</td>
      <td>['Action', 'Adventure', 'Drama', 'Fantasy', 'M...</td>
      <td>24.0</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>8.941080</td>
      <td>3</td>
      <td>3331144</td>
      <td>225215</td>
      <td>After a horrific alchemy experiment goes wrong...</td>
      <td>['Park, Romi', 'Kugimiya, Rie', 'Miki, Shinich...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9253</td>
      <td>Steins;Gate</td>
      <td>TV</td>
      <td>24.0</td>
      <td>['Frontier Works', 'Media Factory', 'Kadokawa ...</td>
      <td>['Funimation']</td>
      <td>['White Fox']</td>
      <td>['Drama', 'Sci-Fi', 'Suspense', 'Psychological...</td>
      <td>24.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.962588</td>
      <td>13</td>
      <td>2553356</td>
      <td>189031</td>
      <td>Eccentric scientist Rintarou Okabe has a never...</td>
      <td>['Miyano, Mamoru', 'Imai, Asami', 'Hanazawa, K...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>28977</td>
      <td>Gintama°</td>
      <td>TV</td>
      <td>51.0</td>
      <td>['TV Tokyo', 'Aniplex', 'Dentsu']</td>
      <td>['Funimation', 'Crunchyroll']</td>
      <td>['Bandai Namco Pictures']</td>
      <td>['Action', 'Comedy', 'Sci-Fi', 'Gag Humor', 'H...</td>
      <td>24.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.726812</td>
      <td>341</td>
      <td>628071</td>
      <td>16610</td>
      <td>Gintoki, Shinpachi, and Kagura return as the f...</td>
      <td>['Sugita, Tomokazu', 'Kugimiya, Rie', 'Sakaguc...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38524</td>
      <td>Shingeki no Kyojin Season 3 Part 2</td>
      <td>TV</td>
      <td>10.0</td>
      <td>['Production I.G', 'Dentsu', 'Mainichi Broadca...</td>
      <td>['Funimation']</td>
      <td>['Wit Studio']</td>
      <td>['Action', 'Drama', 'Suspense', 'Gore', 'Milit...</td>
      <td>23.0</td>
      <td>R - 17+ (violence &amp; profanity)</td>
      <td>9.019487</td>
      <td>21</td>
      <td>2262916</td>
      <td>58383</td>
      <td>Seeking to restore humanity's diminishing hope...</td>
      <td>['Kamiya, Hiroshi', 'Kaji, Yuuki', 'Ishikawa, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Initialise our content based recommender object and evaluator object
content_rec = ContentBasedRecommender(df_content)
tester_eval = ModelEvaluator()
```


```python
# No multiprocessing
s = datetime.now()
test_input_df, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==47], random_state=1) 
pred_df = content_rec.predict(test_input_df, 10)
print("Final MRR: " , tester_eval.evaluate_mrr(pred_df, test_input_df, test_val_df, topn=10))
print("Final NDCG: " , tester_eval.evaluate_ndcg(pred_df, test_input_df, test_val_df, topn=10))
print(datetime.now()-s)
```

    Final MRR:  1.0
    Final NDCG:  0.8332242176357783
    0:00:01.134000
    


```python
pred_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Preds</th>
      <th>Weights</th>
      <th>MAL_Id</th>
      <th>Name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Producers</th>
      <th>Licensors</th>
      <th>Studios</th>
      <th>Genres</th>
      <th>Duration</th>
      <th>Rating</th>
      <th>Score</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Synopsis</th>
      <th>Voice_Actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>117</th>
      <td>38474</td>
      <td>0.771254</td>
      <td>38474</td>
      <td>Yuru Camp△ Season 2</td>
      <td>TV</td>
      <td>13.0</td>
      <td>['Half H.P Studio', 'MAGES.', 'DeNA']</td>
      <td>['None found', 'add some']</td>
      <td>['C-Station']</td>
      <td>['Slice of Life', 'CGDCT', 'Iyashikei']</td>
      <td>23.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.504338</td>
      <td>1079</td>
      <td>222123</td>
      <td>3039</td>
      <td>Having spent Christmas camping with her new fr...</td>
      <td>['Touyama, Nao', 'Hanamori, Yumiri', 'Toyosaki...</td>
    </tr>
    <tr>
      <th>1593</th>
      <td>54005</td>
      <td>0.706282</td>
      <td>54005</td>
      <td>COLORs</td>
      <td>ONA</td>
      <td>1.0</td>
      <td>['TOHO animation']</td>
      <td>['None found', 'add some']</td>
      <td>['Wit Studio']</td>
      <td>['Drama', 'Crossdressing']</td>
      <td>3.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>7.616992</td>
      <td>7294</td>
      <td>7142</td>
      <td>50</td>
      <td>A girl finds herself mesmerized by a young wom...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1651</th>
      <td>37341</td>
      <td>0.691259</td>
      <td>37341</td>
      <td>Yuru Camp△ Specials</td>
      <td>Special</td>
      <td>3.0</td>
      <td>['None found', 'add some']</td>
      <td>['None found', 'add some']</td>
      <td>['C-Station']</td>
      <td>['Slice of Life', 'CGDCT', 'Iyashikei']</td>
      <td>8.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>7.581927</td>
      <td>2934</td>
      <td>55632</td>
      <td>90</td>
      <td>When Chiaki Oogaki and Aoi Inuyama start the O...</td>
      <td>['Touyama, Nao', 'Hanamori, Yumiri', 'Toyosaki...</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>51958</td>
      <td>0.637903</td>
      <td>51958</td>
      <td>Kono Subarashii Sekai ni Bakuen wo!</td>
      <td>TV</td>
      <td>12.0</td>
      <td>['Half H.P Studio', 'Nippon Columbia', 'Atelie...</td>
      <td>['None found', 'add some']</td>
      <td>['Drive']</td>
      <td>['Comedy', 'Fantasy']</td>
      <td>23.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>7.461288</td>
      <td>768</td>
      <td>309112</td>
      <td>1725</td>
      <td>Megumin is a young and passionate wizard from ...</td>
      <td>['Takahashi, Rie', 'Toyosaki, Aki', 'Fukushima...</td>
    </tr>
    <tr>
      <th>316</th>
      <td>53888</td>
      <td>0.580874</td>
      <td>53888</td>
      <td>Spy x Family Movie: Code: White</td>
      <td>Movie</td>
      <td>1.0</td>
      <td>['TOHO animation', 'Shueisha']</td>
      <td>['None found', 'add some']</td>
      <td>['Wit Studio', 'CloverWorks']</td>
      <td>['Action', 'Comedy', 'Childcare', 'Shounen']</td>
      <td>110.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.358056</td>
      <td>2046</td>
      <td>101970</td>
      <td>335</td>
      <td>After receiving an order to be replaced in Ope...</td>
      <td>['Tanezaki, Atsumi', 'Hayami, Saori', 'Eguchi,...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Multiprocessing
s = datetime.now()
test_input_df, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==47], random_state=1) 
pred_df = content_rec.par_predict(test_input_df, 10)
print("Final MRR: " , tester_eval.evaluate_mrr(pred_df, test_input_df, test_val_df, topn=10))
print("Final NDCG: " , tester_eval.evaluate_ndcg(pred_df, test_input_df, test_val_df, topn=10))
print(datetime.now()-s)
```

    Final MRR:  1.0
    Final NDCG:  0.8332242176357783
    0:00:00.720000
    


```python
pred_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Preds</th>
      <th>Weights</th>
      <th>MAL_Id</th>
      <th>Name</th>
      <th>Type</th>
      <th>Episodes</th>
      <th>Producers</th>
      <th>Licensors</th>
      <th>Studios</th>
      <th>Genres</th>
      <th>Duration</th>
      <th>Rating</th>
      <th>Score</th>
      <th>Popularity</th>
      <th>Members</th>
      <th>Favorites</th>
      <th>Synopsis</th>
      <th>Voice_Actors</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>117</th>
      <td>38474</td>
      <td>0.771254</td>
      <td>38474</td>
      <td>Yuru Camp△ Season 2</td>
      <td>TV</td>
      <td>13.0</td>
      <td>['Half H.P Studio', 'MAGES.', 'DeNA']</td>
      <td>['None found', 'add some']</td>
      <td>['C-Station']</td>
      <td>['Slice of Life', 'CGDCT', 'Iyashikei']</td>
      <td>23.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.504338</td>
      <td>1079</td>
      <td>222123</td>
      <td>3039</td>
      <td>Having spent Christmas camping with her new fr...</td>
      <td>['Touyama, Nao', 'Hanamori, Yumiri', 'Toyosaki...</td>
    </tr>
    <tr>
      <th>1593</th>
      <td>54005</td>
      <td>0.706282</td>
      <td>54005</td>
      <td>COLORs</td>
      <td>ONA</td>
      <td>1.0</td>
      <td>['TOHO animation']</td>
      <td>['None found', 'add some']</td>
      <td>['Wit Studio']</td>
      <td>['Drama', 'Crossdressing']</td>
      <td>3.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>7.616992</td>
      <td>7294</td>
      <td>7142</td>
      <td>50</td>
      <td>A girl finds herself mesmerized by a young wom...</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1651</th>
      <td>37341</td>
      <td>0.691259</td>
      <td>37341</td>
      <td>Yuru Camp△ Specials</td>
      <td>Special</td>
      <td>3.0</td>
      <td>['None found', 'add some']</td>
      <td>['None found', 'add some']</td>
      <td>['C-Station']</td>
      <td>['Slice of Life', 'CGDCT', 'Iyashikei']</td>
      <td>8.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>7.581927</td>
      <td>2934</td>
      <td>55632</td>
      <td>90</td>
      <td>When Chiaki Oogaki and Aoi Inuyama start the O...</td>
      <td>['Touyama, Nao', 'Hanamori, Yumiri', 'Toyosaki...</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>51958</td>
      <td>0.637903</td>
      <td>51958</td>
      <td>Kono Subarashii Sekai ni Bakuen wo!</td>
      <td>TV</td>
      <td>12.0</td>
      <td>['Half H.P Studio', 'Nippon Columbia', 'Atelie...</td>
      <td>['None found', 'add some']</td>
      <td>['Drive']</td>
      <td>['Comedy', 'Fantasy']</td>
      <td>23.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>7.461288</td>
      <td>768</td>
      <td>309112</td>
      <td>1725</td>
      <td>Megumin is a young and passionate wizard from ...</td>
      <td>['Takahashi, Rie', 'Toyosaki, Aki', 'Fukushima...</td>
    </tr>
    <tr>
      <th>316</th>
      <td>53888</td>
      <td>0.580874</td>
      <td>53888</td>
      <td>Spy x Family Movie: Code: White</td>
      <td>Movie</td>
      <td>1.0</td>
      <td>['TOHO animation', 'Shueisha']</td>
      <td>['None found', 'add some']</td>
      <td>['Wit Studio', 'CloverWorks']</td>
      <td>['Action', 'Comedy', 'Childcare', 'Shounen']</td>
      <td>110.0</td>
      <td>PG-13 - Teens 13 or older</td>
      <td>8.358056</td>
      <td>2046</td>
      <td>101970</td>
      <td>335</td>
      <td>After receiving an order to be replaced in Ope...</td>
      <td>['Tanezaki, Atsumi', 'Hayami, Saori', 'Eguchi,...</td>
    </tr>
  </tbody>
</table>
</div>



Sanity check on the same User Id 47 above shows that both calculations from parallel/non-parallel processing functions are the same, and the parallel calculations results in about 50% less computation time.


```python
count, total_mrr, total_ndcg = 0, 0, 0
s = datetime.now()
number_of_samples = 1000
mrr_content, ndcg_content= [], []
for i in np.random.choice(df_ratings.User_Id.unique(), number_of_samples, replace=False):
    s_inner = datetime.now()
    count += 1
    test_input_df, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==i], random_state=1) 
    pred_df = content_rec.par_predict(test_input_df, 10)
    mrr =  tester_eval.evaluate_mrr(pred_df, test_input_df, test_val_df, topn=10)
    ndcg = tester_eval.evaluate_ndcg(pred_df, test_input_df, test_val_df, topn=10)
    total_mrr += mrr
    total_ndcg += ndcg
    mrr_content.append(mrr)
    ndcg_content.append(ndcg)
```

As the computational time required is significant higher when compared to our base model, we will evaluate our subsequent models on a subset of 380 out of the 16744 users we have in our dataset to obtain results with 95% confidence interval at 5% margin of error.


```python
print(f'Content MRR: {running_avg(mrr_content)[-1]}')
print(f'Content NDCG: {running_avg(ndcg_content)[-1]}')
```

    Content MRR: 0.6794964285714287
    Content NDCG: 0.6826560139876724
    

We see that our content based recommendation system barely beats our baseline model.

<a id='5'></a>
## 5. Collaborative Filtering
Within this section we will utilise preferences and ratings information from many users to create predictions on what other similar users may be interested in.

The underlying assaumption here is that users with similar preferences and opinions would prefer the same titles as one another.

To make recommendations, the user's input data will be appended to our ratings data and singular value decomposition (SVD) will be applied to factorize the matrix. Thereafter a dot product of the feature vector corresponding to the input user with the feature vectors corresponding to the titles will return similarity measures that we can use to make recommendations.

The same process can be applied to inputs with multiple users, this can be useful for making recommendations to groups of people looking for new titles to watch together.


```python
df_ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Username</th>
      <th>User_Id</th>
      <th>Anime_Id</th>
      <th>Anime_Title</th>
      <th>Rating_Status</th>
      <th>Rating_Score</th>
      <th>Num_Epi_Watched</th>
      <th>Is_Rewatching</th>
      <th>Updated</th>
      <th>Start_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>flerbz</td>
      <td>0</td>
      <td>30654</td>
      <td>Ansatsu Kyoushitsu 2nd Season</td>
      <td>watching</td>
      <td>0</td>
      <td>24</td>
      <td>False</td>
      <td>2022-02-26 22:15:01+00:00</td>
      <td>2022-01-29</td>
    </tr>
    <tr>
      <th>1</th>
      <td>flerbz</td>
      <td>0</td>
      <td>22789</td>
      <td>Barakamon</td>
      <td>dropped</td>
      <td>0</td>
      <td>2</td>
      <td>False</td>
      <td>2023-01-28 19:03:33+00:00</td>
      <td>2022-04-06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>flerbz</td>
      <td>0</td>
      <td>31964</td>
      <td>Boku no Hero Academia</td>
      <td>completed</td>
      <td>0</td>
      <td>13</td>
      <td>False</td>
      <td>2024-03-31 02:10:32+00:00</td>
      <td>2024-03-30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>flerbz</td>
      <td>0</td>
      <td>33486</td>
      <td>Boku no Hero Academia 2nd Season</td>
      <td>completed</td>
      <td>0</td>
      <td>25</td>
      <td>False</td>
      <td>2024-03-31 22:32:02+00:00</td>
      <td>2024-03-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>flerbz</td>
      <td>0</td>
      <td>36456</td>
      <td>Boku no Hero Academia 3rd Season</td>
      <td>watching</td>
      <td>0</td>
      <td>24</td>
      <td>False</td>
      <td>2024-04-03 02:08:56+00:00</td>
      <td>2024-03-31</td>
    </tr>
  </tbody>
</table>
</div>




```python
def pivot_ratings(df):
    df['Mean_Score'] = 0
    mean_df = df[df['Rating_Score']>0].groupby("User_Id")['Rating_Score'].mean().reset_index().rename(columns={'Rating_Score':'mean_score'})
    df = df.merge(mean_df)
    df['Interactions'] = 0.0
    df.loc[df.Rating_Score == 0, 'Interactions'] = 2
    df.loc[df.Rating_Score-df.Mean_Score < 0, 'Interactions'] = 1
    df.loc[df.Rating_Score-df.Mean_Score == 0, 'Interactions'] = 3
    df.loc[df.Rating_Score-df.Mean_Score > 0, 'Interactions'] = 4
    df = df.pivot(index='User_Id', columns='Anime_Id', values='Interactions').fillna(0)
    return df
```

The above function calculates the mean ratings per rated title for each user, and subtracts this mean from all of the ratings the user has made to remove rating biases. An interaction score is then computed based on how well the user rated the interaction. A pivot is then applied to the dataframe preparing it for modeling.


```python
df_cf = df_ratings.copy()
df_cf = pivot_ratings(df_cf)
df_cf.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Anime_Id</th>
      <th>1</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>...</th>
      <th>58564</th>
      <th>58567</th>
      <th>58569</th>
      <th>58572</th>
      <th>58573</th>
      <th>58592</th>
      <th>58600</th>
      <th>58603</th>
      <th>58614</th>
      <th>58632</th>
    </tr>
    <tr>
      <th>User_Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 17178 columns</p>
</div>




```python
df_cf.shape
```




    (15615, 17178)




```python
test_input_df_original, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==47], random_state=1) 
test_input_df = pivot_ratings(test_input_df_original)
test_input_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Anime_Id</th>
      <th>1</th>
      <th>5</th>
      <th>32</th>
      <th>4037</th>
      <th>11757</th>
      <th>14719</th>
      <th>30831</th>
      <th>31933</th>
      <th>34798</th>
      <th>38040</th>
      <th>...</th>
      <th>49026</th>
      <th>50265</th>
      <th>50602</th>
      <th>50710</th>
      <th>51179</th>
      <th>52701</th>
      <th>52741</th>
      <th>53887</th>
      <th>54829</th>
      <th>55818</th>
    </tr>
    <tr>
      <th>User_Id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>




```python
# calculating new index labels for the test input
new_index = pd.Series(list(range(df_cf.index[-1] + 1, df_cf.index[-1] + 1 + len(test_input_df))))
new_index
```




    0    20011
    dtype: int64




```python
# Ratings dataset + user input data
df_cf = pd.concat([df_cf, test_input_df.set_index(new_index)]).fillna(0)
df_cf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Anime_Id</th>
      <th>1</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>...</th>
      <th>58564</th>
      <th>58567</th>
      <th>58569</th>
      <th>58572</th>
      <th>58573</th>
      <th>58592</th>
      <th>58600</th>
      <th>58603</th>
      <th>58614</th>
      <th>58632</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20007</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20008</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20009</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20010</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>20011</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>15616 rows × 17178 columns</p>
</div>




```python
# Applying SVD on sparse matrix
s = datetime.now()
sparse_cf = csr_matrix(df_cf)
U, sigma, Vt = svds(sparse_cf)
print(datetime.now() - s)
```

    0:00:04.596998
    


```python
U.shape
```




    (15616, 6)




```python
Vt.shape
```




    (6, 17178)




```python
sigma = np.diag(sigma)
```


```python
# Reconstruct matrix and normalizing the measures
all_ratings = np.dot(np.dot(U, sigma), Vt)
all_ratings = (all_ratings - all_ratings.min()) / (all_ratings.max() - all_ratings.min())
all_ratings.shape
```




    (15616, 17178)




```python
# Show our reconstructed matrix, for each row of user_id the columns show how closely aligned they are with each anime title
df_cf_pred = pd.DataFrame(all_ratings, columns = df_cf.columns, index = df_cf.index)
df_cf_pred
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Anime_Id</th>
      <th>1</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>...</th>
      <th>58564</th>
      <th>58567</th>
      <th>58569</th>
      <th>58572</th>
      <th>58573</th>
      <th>58592</th>
      <th>58600</th>
      <th>58603</th>
      <th>58614</th>
      <th>58632</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.322143</td>
      <td>0.281029</td>
      <td>0.284811</td>
      <td>0.265056</td>
      <td>0.264715</td>
      <td>0.270996</td>
      <td>0.261508</td>
      <td>0.264742</td>
      <td>0.271192</td>
      <td>0.309478</td>
      <td>...</td>
      <td>0.264124</td>
      <td>0.276174</td>
      <td>0.264103</td>
      <td>0.270113</td>
      <td>0.268170</td>
      <td>0.265395</td>
      <td>0.264182</td>
      <td>0.264130</td>
      <td>0.264266</td>
      <td>0.264123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.426405</td>
      <td>0.302308</td>
      <td>0.289836</td>
      <td>0.264833</td>
      <td>0.262356</td>
      <td>0.255552</td>
      <td>0.300901</td>
      <td>0.262838</td>
      <td>0.282760</td>
      <td>0.416440</td>
      <td>...</td>
      <td>0.264126</td>
      <td>0.313230</td>
      <td>0.264115</td>
      <td>0.285468</td>
      <td>0.267381</td>
      <td>0.268286</td>
      <td>0.267278</td>
      <td>0.264312</td>
      <td>0.264614</td>
      <td>0.264151</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.355228</td>
      <td>0.353592</td>
      <td>0.215732</td>
      <td>0.261275</td>
      <td>0.276188</td>
      <td>0.306569</td>
      <td>0.335670</td>
      <td>0.266856</td>
      <td>0.261291</td>
      <td>0.128861</td>
      <td>...</td>
      <td>0.264151</td>
      <td>0.243649</td>
      <td>0.264216</td>
      <td>0.253360</td>
      <td>0.270926</td>
      <td>0.263537</td>
      <td>0.269985</td>
      <td>0.265600</td>
      <td>0.264068</td>
      <td>0.264374</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.335182</td>
      <td>0.252427</td>
      <td>0.282612</td>
      <td>0.265102</td>
      <td>0.265241</td>
      <td>0.278636</td>
      <td>0.267611</td>
      <td>0.264956</td>
      <td>0.287778</td>
      <td>0.306845</td>
      <td>...</td>
      <td>0.264129</td>
      <td>0.291591</td>
      <td>0.264180</td>
      <td>0.277689</td>
      <td>0.264350</td>
      <td>0.267018</td>
      <td>0.264216</td>
      <td>0.264182</td>
      <td>0.264060</td>
      <td>0.264139</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.196523</td>
      <td>0.209384</td>
      <td>0.235906</td>
      <td>0.261257</td>
      <td>0.262673</td>
      <td>0.249039</td>
      <td>0.244840</td>
      <td>0.261336</td>
      <td>0.264960</td>
      <td>0.233427</td>
      <td>...</td>
      <td>0.264131</td>
      <td>0.296764</td>
      <td>0.264170</td>
      <td>0.282507</td>
      <td>0.261342</td>
      <td>0.268099</td>
      <td>0.265622</td>
      <td>0.264338</td>
      <td>0.264221</td>
      <td>0.264163</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20007</th>
      <td>0.489428</td>
      <td>0.306336</td>
      <td>0.338105</td>
      <td>0.269412</td>
      <td>0.263843</td>
      <td>0.272239</td>
      <td>0.307359</td>
      <td>0.264999</td>
      <td>0.296388</td>
      <td>0.462530</td>
      <td>...</td>
      <td>0.264125</td>
      <td>0.293219</td>
      <td>0.264164</td>
      <td>0.272673</td>
      <td>0.261987</td>
      <td>0.265958</td>
      <td>0.264268</td>
      <td>0.264092</td>
      <td>0.264209</td>
      <td>0.264122</td>
    </tr>
    <tr>
      <th>20008</th>
      <td>0.543922</td>
      <td>0.333727</td>
      <td>0.363319</td>
      <td>0.268517</td>
      <td>0.266733</td>
      <td>0.287688</td>
      <td>0.256264</td>
      <td>0.265529</td>
      <td>0.297220</td>
      <td>0.494144</td>
      <td>...</td>
      <td>0.264127</td>
      <td>0.331264</td>
      <td>0.264056</td>
      <td>0.296463</td>
      <td>0.279677</td>
      <td>0.271361</td>
      <td>0.265253</td>
      <td>0.264365</td>
      <td>0.264831</td>
      <td>0.264159</td>
    </tr>
    <tr>
      <th>20009</th>
      <td>0.485117</td>
      <td>0.329416</td>
      <td>0.334906</td>
      <td>0.268958</td>
      <td>0.264527</td>
      <td>0.280297</td>
      <td>0.299506</td>
      <td>0.266346</td>
      <td>0.291095</td>
      <td>0.443311</td>
      <td>...</td>
      <td>0.264120</td>
      <td>0.283117</td>
      <td>0.264103</td>
      <td>0.269180</td>
      <td>0.268408</td>
      <td>0.264956</td>
      <td>0.263968</td>
      <td>0.263980</td>
      <td>0.264310</td>
      <td>0.264098</td>
    </tr>
    <tr>
      <th>20010</th>
      <td>0.602587</td>
      <td>0.379842</td>
      <td>0.355045</td>
      <td>0.271216</td>
      <td>0.264315</td>
      <td>0.282937</td>
      <td>0.351978</td>
      <td>0.267386</td>
      <td>0.300561</td>
      <td>0.525480</td>
      <td>...</td>
      <td>0.264118</td>
      <td>0.276331</td>
      <td>0.264112</td>
      <td>0.261242</td>
      <td>0.266583</td>
      <td>0.263021</td>
      <td>0.264786</td>
      <td>0.263954</td>
      <td>0.264307</td>
      <td>0.264092</td>
    </tr>
    <tr>
      <th>20011</th>
      <td>0.281417</td>
      <td>0.267700</td>
      <td>0.271291</td>
      <td>0.264485</td>
      <td>0.263932</td>
      <td>0.262970</td>
      <td>0.264506</td>
      <td>0.263964</td>
      <td>0.265495</td>
      <td>0.284469</td>
      <td>...</td>
      <td>0.264124</td>
      <td>0.268596</td>
      <td>0.264120</td>
      <td>0.266007</td>
      <td>0.264448</td>
      <td>0.264542</td>
      <td>0.264210</td>
      <td>0.264130</td>
      <td>0.264178</td>
      <td>0.264125</td>
    </tr>
  </tbody>
</table>
<p>15616 rows × 17178 columns</p>
</div>




```python
test_pred = df_cf_pred.loc[20011].sort_values(ascending=False).reset_index()
test_pred = test_pred.loc[~test_pred['Anime_Id'].isin(test_input_df_original['Anime_Id'])]
test_pred.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anime_Id</th>
      <th>20011</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>16498</td>
      <td>0.308918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25777</td>
      <td>0.306242</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35760</td>
      <td>0.306020</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38524</td>
      <td>0.305755</td>
    </tr>
    <tr>
      <th>5</th>
      <td>40028</td>
      <td>0.305438</td>
    </tr>
  </tbody>
</table>
</div>




```python
tester_eval = ModelEvaluator()
print('MRR : ', tester_eval.evaluate_mrr(test_pred, None, test_val_df, topn=10, left_on='Anime_Id'))
print('NDCG : ', tester_eval.evaluate_ndcg(test_pred, None, test_val_df, topn=10, left_on='Anime_Id'))
```

    MRR :  0.14285714285714285
    NDCG :  0.32166167872792356
    


```python
# User only interacted with 1 title out of the top 10 predictions
test_pred.iloc[:10].merge(test_val_df, how='left')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Anime_Id</th>
      <th>20011</th>
      <th>Username</th>
      <th>User_Id</th>
      <th>Anime_Title</th>
      <th>Rating_Status</th>
      <th>Rating_Score</th>
      <th>Num_Epi_Watched</th>
      <th>Is_Rewatching</th>
      <th>Updated</th>
      <th>Start_Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16498</td>
      <td>0.308918</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25777</td>
      <td>0.306242</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35760</td>
      <td>0.306020</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38524</td>
      <td>0.305755</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40028</td>
      <td>0.305438</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>40748</td>
      <td>0.304172</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>38000</td>
      <td>0.304036</td>
      <td>Pynkmouth</td>
      <td>47.0</td>
      <td>Kimetsu no Yaiba</td>
      <td>plan_to_watch</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>False</td>
      <td>2021-03-18 22:08:00+00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>48583</td>
      <td>0.302796</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>47778</td>
      <td>0.301383</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>52991</td>
      <td>0.301243</td>
      <td>Pynkmouth</td>
      <td>47.0</td>
      <td>Sousou no Frieren</td>
      <td>completed</td>
      <td>6.0</td>
      <td>28.0</td>
      <td>False</td>
      <td>2024-04-11 10:28:04+00:00</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
class CollaborativeRecommender:
    def __init__(self, df_cf):
        self.df_original = df_cf
        self.df_anime_id = df_cf.groupby(['Anime_Id','Anime_Title']).count().reset_index()[['Anime_Id','Anime_Title']]
               
    def process_df(self, df):
        df['Mean_Score'] = 0
        mean_df = df[df['Rating_Score']>0].groupby("User_Id")['Rating_Score'].mean().reset_index().rename(columns={'Rating_Score':'mean_score'})
        df = df.merge(mean_df)
        df['Interactions'] = 0.0
        df.loc[df.Rating_Score == 0, 'Interactions'] = 2
        df.loc[df.Rating_Score-df.Mean_Score < 0, 'Interactions'] = 1
        df.loc[df.Rating_Score-df.Mean_Score == 0, 'Interactions'] = 3
        df.loc[df.Rating_Score-df.Mean_Score > 0, 'Interactions'] = 4
        df = df.pivot(index='User_Id', columns='Anime_Id', values='Interactions').fillna(0)
        return df
    
    def predict_dec(self, user_df, k=15):
        max_uid = self.df_original.User_Id.max()
        for i, uid in enumerate(user_df.User_Id.unique()):
            user_df.loc[user_df.User_Id==uid, 'User_Id'] = max_uid + 1 + i
        user_df = pd.concat([self.df_original, user_df])
        user_cf = self.process_df(user_df)
        sparse_cf = csr_matrix(user_cf)
        U, sigma, Vt = svds(sparse_cf)
        return U, sigma, Vt, user_cf.columns, user_cf.index
    
    def predict(self, user_df, topn=10, k=15):
        # Reconstruct matrix to find similarities
        U, sigma, Vt, new_col, new_index = self.predict_dec(user_df, k)
        sigma = np.diag(sigma)
        all_ratings = np.dot(np.dot(U,sigma), Vt)
        all_ratings = (all_ratings - all_ratings.min()) / (all_ratings.max() - all_ratings.min())
        
        # Construct output dataframe, collecting weights from the number of user we have predicted on
        df_cf_pred = pd.DataFrame(all_ratings, columns=new_col, index=new_index)     
        num_users = user_df.User_Id.nunique()
        res = df_cf_pred.iloc[-num_users:].T
        if num_users == 1:
            res = res.sort_values(res.columns[0],ascending=False).reset_index()
            res = res.loc[~res['Anime_Id'].isin(user_df['Anime_Id'])][:topn]
        else:
            res = res.reset_index()
            res = res.loc[~res['Anime_Id'].isin(user_df['Anime_Id'])]
        return res
```

As before, we have code to create an object for our Collaborative Recommender System, taking in some inputs and producing recommendations.


```python
# 3 Samples, all anime titles from dataset
df_cf = df_ratings.copy()
cf_rec = CollaborativeRecommender(df_cf)
tester_eval = ModelEvaluator()

count, total_mrr, total_ndcg = 0, 0, 0
s = datetime.now()
number_of_samples = 3
print(f"Number of Anime Titles within our dataset: {df_cf.Anime_Id.nunique()}")
for i in np.random.choice(df_ratings.User_Id.unique(), number_of_samples, replace=False):
    s_inner = datetime.now()
    count += 1
    test_input_df, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==i], random_state=1) 
    pred_df = cf_rec.predict(test_input_df, 10)
    total_mrr += tester_eval.evaluate_mrr(pred_df, test_input_df, test_val_df, topn=10, left_on='Anime_Id')
    total_ndcg += tester_eval.evaluate_ndcg(pred_df, test_input_df, test_val_df, topn=10, left_on='Anime_Id')
    if not count % 10:
        print(f'Time Elapsed : {datetime.now()-s}')
    print(f"Loop Number {count}, User Id {i}, Time Taken {datetime.now()-s_inner}")
print("Final MRR: " , total_mrr/count)
print("Final NDCG: " , total_ndcg/count)
```

    Number of Anime Titles within our dataset: 17365
    Loop Number 1, User Id 13830, Time Taken 0:00:08.383678
    Loop Number 2, User Id 4062, Time Taken 0:00:08.232498
    Loop Number 3, User Id 17416, Time Taken 0:00:08.202000
    Final MRR:  0.7777777777777778
    Final NDCG:  0.8000865280044508
    

Sanity check with 3 samples looks fine, compared to previous approaches this does take a significantly longer time to compute.


```python
df_cf = df_ratings.copy()
cf_rec = CollaborativeRecommender(df_cf)
tester_eval = ModelEvaluator()

count, total_mrr, total_ndcg = 0, 0, 0
s = datetime.now()
number_of_samples = 1000
mrr_collab, ndcg_collab = [], []
for i in np.random.choice(df_ratings.User_Id.unique(), number_of_samples, replace=False):
    s_inner = datetime.now()
    count += 1
    test_input_df, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==i], random_state=1) 
    pred_df = cf_rec.predict(test_input_df, 10)
    mrr = tester_eval.evaluate_mrr(pred_df, test_input_df, test_val_df, topn=10, left_on='Anime_Id')
    ndcg = tester_eval.evaluate_ndcg(pred_df, test_input_df, test_val_df, topn=10, left_on='Anime_Id')
    total_mrr += mrr
    total_ndcg += ndcg
    mrr_collab.append(mrr)
    ndcg_collab.append(ndcg)
```


```python
print(f'Collab MRR: {running_avg(mrr_collab)[-1]}')
print(f'Collab NDCG: {running_avg(ndcg_collab)[-1]}')
```

    Collab MRR: 0.8626269841269844
    Collab NDCG: 0.801510934536861
    

The performance of the collaborative approach is significantly better than the baseline and content base recommndations, suggesting that the assumption that similar users will like similar titles may have some truth in it. A possible explanation for why the performance is not better is that our SVD is not able to accurately recreate the original matrix due to the chosen low number of singular values computed.

Below is a sanity check with 3 samples again, but with obscure titles that have less than 3 ratings removed from the computation

Surprisingly, the performance of this approach is worse than our baseline and content base recommendations. A possible explanation is that our SVD is not able to accurately recreate the original matrix due to the chosen low number of singular values computed. If this approach is re-evaluated with a higher number of singlular values with similarly poor results, it will suggest that the original assumption of similar users liking similar titles is not entire accurate.

Below is a sanity check with 3 samples again, but with obscure titles that have less than 3 ratings removed from the computation


```python
# Remove titles with less than 10 user ratings 
df_ratings_subset = df_ratings[['Anime_Id','Anime_Title']].value_counts().reset_index()
df_ratings_subset = df_ratings_subset[df_ratings_subset['count'] >= 10]
df_cf_subset = df_ratings[df_ratings.Anime_Id.isin(df_ratings_subset.Anime_Id)]
```


```python
print(f"Original number of titles: {df_ratings.Anime_Title.nunique()}")
print(f"Trimmed number of titles: {df_ratings_subset.Anime_Title.nunique()}")
```

    Original number of titles: 17364
    Trimmed number of titles: 10559
    


```python
# 3 Samples, trimmed anime titles from dataset
cf_rec = CollaborativeRecommender(df_cf_subset)
tester_eval = ModelEvaluator()

count, total_mrr, total_ndcg = 0, 0, 0
s = datetime.now()
number_of_samples = 3
print(f"Number of Anime Titles with >= 10 user ratings within our dataset: {df_cf_subset.Anime_Id.nunique()}")
for i in np.random.choice(df_ratings.User_Id.unique(), number_of_samples, replace=False):
    s_inner = datetime.now()
    count += 1
    test_input_df, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==i], random_state=1) 
    pred_df = cf_rec.predict(test_input_df, 10)
    total_mrr += tester_eval.evaluate_mrr(pred_df, test_input_df, test_val_df, topn=10, left_on='Anime_Id')
    total_ndcg += tester_eval.evaluate_ndcg(pred_df, test_input_df, test_val_df, topn=10, left_on='Anime_Id')
    if not count % 10:
        print(f'Time Elapsed : {datetime.now()-s}')
    print(f"Loop Number {count}, User Id {i}, Time Taken {datetime.now()-s_inner}")
print("Final MRR: " , total_mrr/count)
print("Final NDCG: " , total_ndcg/count)
```

    Number of Anime Titles with >= 10 user ratings within our dataset: 10559
    Loop Number 1, User Id 19564, Time Taken 0:00:06.346500
    Loop Number 2, User Id 8772, Time Taken 0:00:06.117500
    Loop Number 3, User Id 16816, Time Taken 0:00:06.155999
    Final MRR:  0.75
    Final NDCG:  0.7081818849660128
    

By trimming the really obscure titles from the dataset we are able to reduce computational times significantly.

<a id='6'></a>
### 6. Hybrid Recommendation
For our hybrid approach we will combine both the Content Based and Collaborative approaches we have explored.

For simplicity's sake we will place a 0.5 weightage for each of the approaches.

Similarity measures from each of these two approaches will be computed separately and standardised so that they are comparable. The final measure will be a combination of the standardised scores based on their weightages.


```python
class HybridRecommender:
    def __init__(self, cb_model, cf_model, df_content, df_ratings, cb_weight=0.5):
        self.cb_model = cb_model(df_content)
        self.cf_model = cf_model(df_ratings)
        self.cb_weight = cb_weight
        self.cf_weight = 1 - cb_weight
        self.n = df_ratings.Anime_Id.nunique()
    
    def predict(self, user_df, topn=10):   
        num_users = user_df.User_Id.nunique()
        cb_pred = self.cb_model.predict(user_df, self.n)
        cf_pred = self.cf_model.predict(user_df, self.n)
        
        # Normalize scores from both predictions
        ss = StandardScaler()
        cb_pred['ss'] = ss.fit_transform(cb_pred['Weights'].values.reshape(-1,1))
        cf_cols = ['ss_' + str(col) for col in cf_pred.columns[-1:]]
        if num_users == 1:
            cf_pred[cf_cols] = ss.fit_transform(cf_pred[cf_pred.columns[-1]].values.reshape(-1,1))
        else:
            cf_pred[cf_cols] = ss.fit_transform(cf_pred[cf_pred.columns[1:]])
        
        combined_pred = cf_pred.merge(cb_pred[['ss','MAL_Id','Name','Score','Popularity']], how='left', left_on='Anime_Id', right_on='MAL_Id')
        combined_pred['Final_Score'] = self.cf_weight*combined_pred[cf_cols].sum(axis=1) + self.cb_weight*combined_pred['ss']
        combined_pred = combined_pred.sort_values('Final_Score', ascending=False)
        return combined_pred[:topn]
```


```python
# Remove titles with less than 10 user ratings 
df_ratings_subset = df_ratings[['Anime_Id','Anime_Title']].value_counts().reset_index()
df_ratings_subset = df_ratings_subset[df_ratings_subset['count'] >= 10]
df_cf_subset = df_ratings[df_ratings.Anime_Id.isin(df_ratings_subset.Anime_Id)]
tester_eval = ModelEvaluator()

hyb_rec = HybridRecommender(ContentBasedRecommender, CollaborativeRecommender, df_content, df_cf_subset, cb_weight = 0.5)
count, total_mrr, total_ndcg = 0, 0, 0
s = datetime.now()
number_of_samples = 1000
print(f"Number of Anime Titles with >= 10 user ratings within our dataset: {df_cf_subset.Anime_Id.nunique()}")
mrr_hybrid, ndcg_hybrid = [], []
for i in np.random.choice(df_ratings.User_Id.unique(), number_of_samples, replace=False):
    s_inner = datetime.now()
    count += 1
    test_input_df, test_val_df = mask_user_ratings(df_ratings[df_ratings['User_Id']==i], random_state=1) 
    if len(test_input_df) == 0:
        continue
    pred_df = hyb_rec.predict(test_input_df, 10)
    mrr = tester_eval.evaluate_mrr(pred_df, test_input_df, test_val_df, topn=10, left_on='Anime_Id')
    ndcg = tester_eval.evaluate_ndcg(pred_df, test_input_df, test_val_df, topn=10, left_on='Anime_Id')
    total_mrr += mrr
    total_ndcg += ndcg
    mrr_hybrid.append(mrr)
    ndcg_hybrid.append(ndcg)
```


```python
print(f'Hybrid MRR: {running_avg(mrr_hybrid)[-1]}')
print(f'Hybrid NDCG: {running_avg(ndcg_hybrid)[-1]}')
```

    Hybrid MRR: 0.9044096472282224
    Hybrid NDCG: 0.8450766763118507
    


```python
mrr_base_trunc = mrr_base[-1000:]
ndcg_base_trunc = ndcg_base[-1000:]
res = [mrr_base_trunc, ndcg_base_trunc, mrr_content,ndcg_content,mrr_collab,ndcg_collab,mrr_hybrid,ndcg_hybrid]
res = [running_avg(scores) for scores in res]
```


```python
# Extract mrr and ndcg from list of all scores
mrr_res = res[::2]
ndcg_res = res[1::2]
```


```python
# Plot averaged cumulative performance over time
fig, ax = plt.subplots(2,1, figsize=(8,8))
mrr_label = ['MRR_Base','MRR_Content','MRR_Collab','MRR_Hybrid']
ndcg_label = ['NDCG_Base','NDCG_Content','NDCG_Collab','NDCG_Hybrid']
for i, r in enumerate(mrr_res):
    sns.lineplot(r, ax = ax[0], label = mrr_label[i])
for i, r in enumerate(ndcg_res):
    sns.lineplot(r, ax = ax[1], label = ndcg_label[i])
plt.xlabel("Samples")
plt.ylabel("Cumulative Average")
plt.suptitle("Cumulative Score Plot")
ax[0].title.set_text("MRR")
ax[1].title.set_text("NDCG")
plt.show()
```


    
![png](/images/mal_rec_exploration/output_93_0.png)
    


The performance of the hybrid approach appears to be significantly better than our previous two approaches and the baseline. This may be due to the limitations mentioned in our previous sections where the content based approach tend to stick to titles of the same kind, and for collaborative approach similar users tend to like similar titles, may not translate well to actual user behaviour. The higher performance of the hybrid approach suggests that in reality users may mainly enjoy titles of the same type while also seeking out some diversity in their interactions, where this diversity coincides with other similar users have interacted with.


```python
final_res = [i[-1] for i in res]
final_mrr = final_res[::2]
final_ndcg = final_res[1::2]
final_results = pd.DataFrame({'MRR':final_mrr, "NDCG":final_ndcg}, index=['Baseline','Content','Collab','Hybrid'])
```


```python
fig,ax = plt.subplots()
c_map = plt.cm.get_cmap('coolwarm').reversed()
im = ax.imshow(final_results.values, cmap=c_map)
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel("Performance (Higher is better)", rotation=-90, va="bottom")

ax.set_xticks(np.arange(final_results.shape[1]), labels=final_results.columns)
ax.set_yticks(np.arange(final_results.shape[0]), labels=final_results.index)
for i in range(final_results.shape[0]):
    for j in range(final_results.shape[1]):
        text = ax.text(j, i, round(final_results.iloc[i, j], 3),
                      ha='center', va='center', color='w')
plt.tight_layout()
plt.title('Comparison of Final Performances')
```




    Text(0.5, 1.0, 'Comparison of Final Performances')




    
![png](/images/mal_rec_exploration/output_96_1.png)
    


Above we see a visual representation of the scores that were seen in this notebook, with the hybrid approach coming out on top compared to our baseline and the other two approaches.

<a id='7'></a>
## 7. Conclusion
In this notebook we have explored different approaches to a recommendation system and discussed some possible limitations and their implications or solutions. For the datasets used, we have shown that the hybrid approach performed best out of the ones tested.

Further improvements can be made to our approaches, some key ones are:
<ul>
<li>- Utilizing the review dataset to provide additional information for our titles</li>
<li>- Improving computational performances of the approaches</li>
<li>- Incorporating additional contextual information such as time to further improve the recommendations</li>
<li>- Exploring more advanced techniques to calculate recommendations </li>
</ul>
