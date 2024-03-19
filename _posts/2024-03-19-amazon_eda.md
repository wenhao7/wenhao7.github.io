---
layout: post
title:  "Exploratory Data Analysis for Amazon's Top 50 Bestselling Books from 2010-2020"
date:   2024-03-19 22:31:08 +0800
category: [data_wrangling, data_analysis, visualization]
tag: [numpy, pandas, seaborn, matplotlib, scipy, webscrape, statistics]
summary: "This notebook explores a dataset containing the top 50 bestselling books on Amazon from the years 2010 to 2020 inclusive. Data was scraped from Amazon webpages and additional information was obtained from Google Books API."
---

***
## *Contents*
1. [Overview](#overview)
2. [Importing Required Libraries](#import)
3. [Data Cleaning](#cleaning)
4. [Analysis](#analysis)
5. [Conclusions](#conclusion)
***

<a id='overview'></a>
## 1. *Overview*
This notebook explores a dataset containing the top 50 bestselling books on Amazon from the years 2010 to 2020 inclusive. Books title, author, rating, number of reviews, price, and year data are scraped from Amazon web pages and genre information is obtained using Google Books API. Webscraping and API calling process can be found in the accompanying file named *'[amazon_scrape.py](https://github.com/wenhao7/Data-Science/blob/main/Amazon%20Books%20EDA/amazon_scrape.py)'*.

***
<a id='import'></a>
## 2. *Importing Required Libraries*


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings("ignore")
sns.set_palette("YlGn")
```

***
<a id='cleaning'></a>
## 3. *Data Cleaning*


```python
# Read data
df = pd.read_csv('Amazon_best_sellers_2010_2020_fiction_flag.csv', encoding='unicode_escape')
df.head()
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
      <th>title</th>
      <th>author</th>
      <th>rating</th>
      <th>reviews</th>
      <th>price</th>
      <th>year</th>
      <th>fiction flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10-Day Green Smoothie Cleanse</td>
      <td>JJ Smith</td>
      <td>4.7</td>
      <td>27719</td>
      <td>1.44</td>
      <td>2016</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11/22/63: A Novel</td>
      <td>Stephen King</td>
      <td>4.7</td>
      <td>2588</td>
      <td>2.81</td>
      <td>2011</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12 Rules for Life: An Antidote to Chaos</td>
      <td>Jordan B. Peterson</td>
      <td>4.7</td>
      <td>39960</td>
      <td>7.31</td>
      <td>2018</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1984 (Signet Classics), Book Cover May Vary</td>
      <td>George Orwell</td>
      <td>4.7</td>
      <td>49411</td>
      <td>0.86</td>
      <td>2017</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5,000 Awesome Facts (About Everything!) (Natio...</td>
      <td>National Kids</td>
      <td>4.8</td>
      <td>15160</td>
      <td>2.50</td>
      <td>2019</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking size of dataset and columns dtypes
print(f'Data contains {df.shape[0]} records and {df.shape[1]} columns.')
df.dtypes
```

    Data contains 546 records and 7 columns.
    




    title            object
    author           object
    rating          float64
    reviews           int64
    price           float64
    year              int64
    fiction flag       bool
    dtype: object




```python
# Change 'fiction flag' column to a categorical input signifying Fiction or Non-Fiction genre
df.loc[df['fiction flag'] == True, 'fiction flag'] = 'Fiction'
df.loc[df['fiction flag'] == False, 'fiction flag'] = 'Non-Fiction'
df['fiction flag'] = df['fiction flag'].astype('category')
df['fiction flag'].dtype
```




    CategoricalDtype(categories=['Fiction', 'Non-Fiction'], ordered=False, categories_dtype=object)




```python
df.head()
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
      <th>title</th>
      <th>author</th>
      <th>rating</th>
      <th>reviews</th>
      <th>price</th>
      <th>year</th>
      <th>fiction flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10-Day Green Smoothie Cleanse</td>
      <td>JJ Smith</td>
      <td>4.7</td>
      <td>27719</td>
      <td>1.44</td>
      <td>2016</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11/22/63: A Novel</td>
      <td>Stephen King</td>
      <td>4.7</td>
      <td>2588</td>
      <td>2.81</td>
      <td>2011</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12 Rules for Life: An Antidote to Chaos</td>
      <td>Jordan B. Peterson</td>
      <td>4.7</td>
      <td>39960</td>
      <td>7.31</td>
      <td>2018</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1984 (Signet Classics), Book Cover May Vary</td>
      <td>George Orwell</td>
      <td>4.7</td>
      <td>49411</td>
      <td>0.86</td>
      <td>2017</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5,000 Awesome Facts (About Everything!) (Natio...</td>
      <td>National Kids</td>
      <td>4.8</td>
      <td>15160</td>
      <td>2.50</td>
      <td>2019</td>
      <td>Non-Fiction</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for any missing data in the dataset
df.isnull().sum()
```




    title           0
    author          0
    rating          0
    reviews         0
    price           0
    year            0
    fiction flag    0
    dtype: int64




```python
# Check for duplicates in 'title' and 'author', ignore 'fiction flag' column as it only contains 'Fiction' and 'Non-Fiction'
for col in ['title','author']:
    if df[col].duplicated().any() == True:
        print(f'Column "{col}" contains duplicates')
    else:
        
        print(f'Column "{col}" contains no duplicates')
```

    Column "title" contains duplicates
    Column "author" contains duplicates
    


```python
# Check for alphabet casing and spacing differences
for col in ['title','author']:
    print(f'"{col}" Original: {len(set(df[col]))}, Edited: {len(set(df[col].str.title().str.strip()))}')

# Make the required edits to standardise book/author names formatting
df.title = df.title.str.title().str.strip()

# Check author names
print(f'Original: {len(df.author.unique())}, Edited: {len(df.author.str.replace(" ","").replace(".","").replace(",","").unique())}')
```

    "title" Original: 345, Edited: 344
    "author" Original: 254, Edited: 254
    Original: 254, Edited: 252
    


```python
# Visually inspect the unique author names present in data set to find duplicates
print(df.author.sort_values().unique())
```

    ['Abraham Verghese' 'Adam Gasiewski' 'Adam Mansbach' 'Adam Wallace'
     'Adir Levy' 'Admiral William H. McRaven' 'Alex Michaelides'
     'Alice Schertle' 'Allie Brosh' 'Amelia Hepworth'
     'American Psychiatric Association' 'American Psychological Association'
     'Amor Towles' 'Amy Ramos' 'Amy Shields' 'Andy Weir' 'Angie Grace'
     'Angie Thomas' 'Ann Voskamp' 'Ann Whitford Paul' 'Anthony Bourdain'
     'Anthony Doerr' 'Atul Gawande' 'B. J. Novak' 'Barack Obama'
     'Bessel van der Kolk M.D.' 'Bill Martin Jr.' "Bill O'Reilly"
     'Blue Star Coloring' 'Bob Woodward' 'Brandon Stanton' 'BreneÌ\x81E Brown'
     'Brian Kilmeade' 'Brit Bennett' 'Bruce Springsteen' 'Carol S. Dweck'
     'Carole P. Roman' 'Celeste Ng' 'Charlaine Harris' 'Charles Duhigg'
     'Charles Krauthammer' 'Charlie Mackesy' 'Cheryl Strayed' 'Chip Gaines'
     'Chip Heath' 'Chris Cleave' 'Chris Kyle' 'Chrissy Teigen'
     'Christina Baker Kline' 'Christopher Paolini' 'Coloring Books for Adults'
     'Conor Riordan' 'Craig Smith' 'Crispin Boyer' 'Crystal Radke' 'DK'
     'Dale Carnegie' 'Dan Brown' 'Daniel James Brown' 'Daniel Kahneman'
     'Daniel Lipkowitz' 'Dav Pilkey' 'Dave Ramsey' 'David Goggins'
     'David Grann' 'David McCullough' 'David Perlmutter MD' 'David Platt'
     'Deborah Diesen' 'Delegates of the Constitutional\x81E\x80¦'
     'Delia Owens' 'Dinah Bucholz' 'Don Miguel Ruiz' 'Donna Tartt'
     'Doug Lemov' 'Dr. Seuss' 'Dr. Steven R Gundry MD' 'Drew Daywalt'
     'E L James' 'Eben Alexander' 'Edward Klein' 'Elie Wiesel'
     'Emily Winfield Martin' 'Eric Carle' 'Eric Larson' 'Erik Larson'
     'Ernest Cline' 'F. Scott Fitzgerald' 'Francis Chan' 'Fredrik Backman'
     'Garth Stein' 'Gary Chapman' 'Gayle Forman' 'Geneen Roth' 'George Orwell'
     'George R. R. Martin' 'George R.R. Martin' 'George W. Bush'
     'Giles Andreae' 'Gillian Flynn' 'Glenn Beck' 'Glennon Doyle'
     'Golden Books' 'Greg Mortenson' 'Harper Lee' 'Hayek' 'Heidi Murkoff'
     'Hillary Rodham Clinton' 'Hopscotch Girls' 'Howard Stern' 'Ian K. Smith'
     'Ibram X. Kendi' 'Ina Garten' 'Isabel Wilkerson' 'J. D. Vance'
     'J. K. Rowling' 'J.K. Rowling' 'JJ Smith' 'James Clear' 'James Comey'
     'James Dashner' 'James Patterson' 'Jay Asher' 'Jaycee Dugard'
     'Jeanine Cummins' 'Jeff Kinney' 'Jen Sincero' 'Jennie Allen'
     'Jennifer Smith' 'Jill Twiss' 'Jim Collins' 'Jim Kay' 'Joanna Gaines'
     'Joel Fuhrman MD' 'Johanna Basford' 'John Bolton' 'John Green'
     'John Grisham' 'John Heilemann' 'Jon Meacham' 'Jon Stewart'
     'Jonathan Cahn' 'Jordan B. Peterson' 'Justin Halpern' 'Kathryn Stockett'
     'Keith Richards' 'Ken Follett' 'Kevin Kwan' 'Khaled Hosseini'
     'Kristin Hannah' 'Larry Schweikart' 'Laura Hillenbrand' 'Laurel Randolph'
     'Lin-Manuel Miranda' 'Lysa TerKeurst' 'M Prefontaine' "Madeleine L'Engle"
     'Malcolm Gladwell' 'Margaret Atwood' 'Margaret Wise Brown'
     'Marie KondÅ\x81E' 'Marjorie Sarnat' 'Mark Hyman M.D.' 'Mark Manson'
     'Mark Owen' 'Mark R. Levin' 'Mark Twain' 'Markus Zusak' 'Marty Noble'
     'Mary L. Trump Ph.D.' 'Matthew McConaughey' 'Melissa Hartwig Urban'
     'Michael Lewis' 'Michael Pollan' 'Michael Wolff' 'Michelle Obama'
     'Mike Moreno' 'Naomi Kleinberg' 'Nathan W. Pyle' 'National Kids'
     'Neil deGrasse Tyson' 'Paper Peony Press' 'Patrick Lencioni'
     'Patrick Thorpe' 'Paul Kalanithi' 'Paula Hawkins' 'Paula McLain'
     'Paulo Coelho' 'Pete Souza' 'Peter A. Lillback' 'Ph.D.' 'Phil Robertson'
     'Pretty Simple Press' 'R. J. Palacio' 'RH Disney' 'Rachel Hollis'
     'Raina Telgemeier' 'Randall Munroe' 'Ray Bradbury' 'Rebecca Skloot'
     'Ree Drummond' 'Rick Riordan' 'Rob Bell' 'Rob Elliott' 'Robert Jordan'
     'Robert Munsch' 'Robin DiAngelo' 'Rod Campbell' 'Roger Priddy'
     'Ron Chernow' 'Rupi Kaur' 'Rush Limbaugh' 'Samin Nosrat' 'Sandra Boynton'
     'Sara Gruen' 'Sarah Young' "Sasha O'Hara" 'Scholastic' 'School Zone'
     'Sean Hannity' 'Shannon Roberts' 'Sharon Jones' 'Sherri Duskey Rinker'
     'Sheryl Sandberg' 'Silly Bear' 'Stephen King' 'Stephen R. Covey'
     'Stephenie Meyer' 'Stieg Larsson' 'Susan Cain' 'Suzanne Collins'
     'Ta-Nehisi Coates' 'Tara Westover' 'Tatiana de Rosnay'
     'The College Board' 'The Staff of The Late Show with\x81E\x80¦'
     'The Washington Post' 'Thomas Campbell' 'Thomas Piketty' 'Thug Kitchen'
     'Timothy Ferriss' 'Tina Fey' 'Todd Burpo' 'Tom Rath' 'Tony Hsieh'
     'Tucker Carlson' 'Veronica Roth' 'Walter Isaacson' 'William Davis'
     'William P. Young' 'Wizards RPG Team' 'Workman Publishing' 'Zhi Gang Sha'
     'no author']
    


```python
# George R.R. Martin and J.K. Rowling appears with two different spellings for their names, standardise to one spelling
df.replace('George R. R. Martin', 'George R.R. Martin', inplace = True)
df.replace('J. K. Rowling', 'J.K. Rowling', inplace = True)
print(f'Original: {len(df.author.unique())}, Edited: {len(df.author.str.replace(" ","").replace(".","").replace(",","").unique())}')
```

    Original: 252, Edited: 252
    


```python
# Check only 2010 - 2020 appear in the dataset
df.year.value_counts()
```




    year
    2018    50
    2017    50
    2019    50
    2020    50
    2015    50
    2013    50
    2012    50
    2016    49
    2011    49
    2014    49
    2010    49
    Name: count, dtype: int64




In this dataset we expect 50 titles present for each year, however from the above we observe that this is not the case as there are years with only 49 titles present. This is due to removed listings for a title within the top 50 Amazon bestsellers in those years preventing the relevant information from being scraped.



```python
df.tail()
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
      <th>title</th>
      <th>author</th>
      <th>rating</th>
      <th>reviews</th>
      <th>price</th>
      <th>year</th>
      <th>fiction flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>541</th>
      <td>Wrecking Ball (Diary Of A Wimpy Kid Book 14)</td>
      <td>Jeff Kinney</td>
      <td>4.9</td>
      <td>16016</td>
      <td>1.74</td>
      <td>2019</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>542</th>
      <td>You Are A Badass: How To Stop Doubting Your Gr...</td>
      <td>Jen Sincero</td>
      <td>4.7</td>
      <td>28561</td>
      <td>1.17</td>
      <td>2019</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>543</th>
      <td>You Are A Badass: How To Stop Doubting Your Gr...</td>
      <td>Jen Sincero</td>
      <td>4.7</td>
      <td>28561</td>
      <td>1.17</td>
      <td>2018</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>544</th>
      <td>You Are A Badass: How To Stop Doubting Your Gr...</td>
      <td>Jen Sincero</td>
      <td>4.7</td>
      <td>28561</td>
      <td>1.17</td>
      <td>2017</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>545</th>
      <td>You Are A Badass: How To Stop Doubting Your Gr...</td>
      <td>Jen Sincero</td>
      <td>4.7</td>
      <td>28561</td>
      <td>1.17</td>
      <td>2016</td>
      <td>Fiction</td>
    </tr>
  </tbody>
</table>
</div>



We can observe that there are duplicates within the dataset if a title makes it to the top50 in different years. The scraped rating, reviews, and price data are the latest values as of scraping, not the values from the particular year the title made it to the top50. Hence, we will create a separate dataframe removing all the duplicated titles to supplement our analysis.


```python
# Separate dataframe containing only unique titles
df_no_dup = df.drop_duplicates('title').reset_index().drop('index', axis = 1)
print(f'Data contains {len(df_no_dup)} books written by {len(df_no_dup.author.unique())} different authors')
```

    Data contains 344 books written by 252 different authors
    

***
<a id='analysis'></a>
# 3. *Analysis*
In this section we will analyse the data and answer a few simple questions about the dataset:<br>
a. [Which author has the highest average rating?](#author-rating)<br>
b. [Which author has the most bestsellers?](#author-title)<br>
c. [Which book has the highest number of reviews?](#author-reviews)<br>
d. [Are ratings, number of reviews, prices, and genre correlated?](#correlation)<br>
e. [Are the distribution of ratings for Fiction and Non-Fiction books the same?](#testing)<br>

This notebook will not be exploring the changes to the books' statistics throughout the years as the dataset only contains the latest statistics as seen in the previous section with Jen Sincero's bestseller 'You Are A Badass'.

<a id='author-rating'></a>
### a. *Which author has the highest average rating?*
When considering highest average ratings, we can look from it from two different angles:<br>
(i) Highest average rating with any number of bestsellers<br>
(ii) Highest average rating with a minimum number of bestsellers (for this analysis we will arbitrarily select authors with a minimum of 3 bestsellers)<br>

By analysing the data in this manner, we can see a list of top authors that may have highly rated 'one-hit wonders', and a list of top authors that have released multiple bestsellers that are more consistently highly rated.


```python
# (i) Highest average rating for authors with any number of bestsellers
top_authors = df.groupby('author').agg(count=('author','size'), mean_rating=('rating','mean')).sort_values('mean_rating', ascending=False).reset_index()
top_authors.head()
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
      <th>author</th>
      <th>count</th>
      <th>mean_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dav Pilkey</td>
      <td>8</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lin-Manuel Miranda</td>
      <td>1</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mark R. Levin</td>
      <td>1</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Patrick Thorpe</td>
      <td>1</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pete Souza</td>
      <td>1</td>
      <td>4.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
# (ii) Highest average rating for authors with at least 3 bestsellers
top_authors = df.groupby('author').agg(count=('author','size'), mean_rating=('rating','mean'))
top_authors = top_authors.loc[top_authors['count']>3].sort_values(['mean_rating','count'], ascending=False).reset_index()
top_authors.head()
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
      <th>author</th>
      <th>count</th>
      <th>mean_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dav Pilkey</td>
      <td>8</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Eric Carle</td>
      <td>8</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sarah Young</td>
      <td>6</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Bill Martin Jr.</td>
      <td>4</td>
      <td>4.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Emily Winfield Martin</td>
      <td>4</td>
      <td>4.9</td>
    </tr>
  </tbody>
</table>
</div>



For both cases we see that there are overlaps in the authors with highest average ratings in both cases, and the top authors all have an average rating of 4.9.

<a id='author-title'></a>
### b. *Which author has the most bestsellers?*
Similarly, in this case we can look at this question from two perspective:<br>
(i) Authors that has made it to the bestselling list the most times<br>
(ii) Authors that has the most number of unique titles in the bestselling list<br>

By analysing the data in this manner we can obtain separate the lists of authors who has made it to the bestselling lists the most times, and the lists of authors who has written the most bestsellers.


```python
# (i) Authors that has made it to the bestselling list the most times
dict_appearance = df.author.value_counts().to_dict()
number_of_appearances = sorted(dict_appearance.items(), key = lambda x:x[1], reverse = True)
x = [number_of_appearances[i][0] for i in range(10)]
y = [number_of_appearances[i][1] for i in range(10)]
```


```python
sns.barplot(x=x, y=y, palette="YlGn")
plt.title('Top 10 Authors With Most Appearances In Top 50 Bestsellers')
plt.xticks(rotation=45, horizontalalignment='right')
plt.ylabel('No. of Appearances')
plt.xlabel('Author')
```




    Text(0.5, 0, 'Author')




    
![png](/images/amazon_eda/output_22_1.png)
    



```python
# (ii) Authors that has the most number of unique titles in the bestselling list
dict_unique_books = df_no_dup.author.value_counts().to_dict()
number_of_unique_books = sorted(dict_unique_books.items(), key = lambda x:x[1], reverse = True) # compare to previous list authors like Jeff Kinney have bestselling books that appear in top50 for a year while Suzanne Collins have books that appear in multiple years

x = [number_of_unique_books[i][0] for i in range(10)]
y = [number_of_unique_books[i][1] for i in range(10)]
```


```python
sns.barplot(x=x, y=y, palette="YlGn")
plt.title('Top 10 Authors With Most Unique Titles In Top 50 Bestsellers')
plt.xticks(rotation=45, horizontalalignment='right')
plt.ylabel('No. of Unique Titles')
plt.xlabel('Author')
```




    Text(0.5, 0, 'Author')




    
![png](/images/amazon_eda/output_24_1.png)
    


In scenario (i) we see that Suzanne Collins has appeared 12 times while Jeff Kinney has appeared 11 times in the bestselling lists, while in scenario (ii) we see that Suzanne Collins has 6 unique titles while Jeff Kinney has 11 unique titles in the bestselling lists. This suggests that Jeff Kinney's bestsellers are popular for their respective bestselling years while Suzanne Collins' bestsellers may be popular for a longer period of time, with some titles appearing in the bestselling lists for multiple years.

<a id='author-reviews'></a>
### c. *Which book has the highest number of reviews?*


```python
df_no_dup.sort_values('reviews', ascending = False).reset_index().head(20)
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
      <th>index</th>
      <th>title</th>
      <th>author</th>
      <th>rating</th>
      <th>reviews</th>
      <th>price</th>
      <th>year</th>
      <th>fiction flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77</td>
      <td>Educated: A Memoir</td>
      <td>Tara Westover</td>
      <td>4.6</td>
      <td>1697195</td>
      <td>19.60</td>
      <td>2019</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>1</th>
      <td>335</td>
      <td>Where The Crawdads Sing</td>
      <td>Delia Owens</td>
      <td>4.6</td>
      <td>1697195</td>
      <td>2.32</td>
      <td>2019</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>2</th>
      <td>325</td>
      <td>Untamed</td>
      <td>Glennon Doyle</td>
      <td>4.6</td>
      <td>1697195</td>
      <td>4.99</td>
      <td>2020</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>3</th>
      <td>295</td>
      <td>The Splendid And The Vile: A Saga Of Churchill...</td>
      <td>Erik Larson</td>
      <td>4.6</td>
      <td>1697195</td>
      <td>7.67</td>
      <td>2020</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>4</th>
      <td>246</td>
      <td>The Girl Who Played With Fire (Millennium Series)</td>
      <td>Stieg Larsson</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>0.02</td>
      <td>2010</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>5</th>
      <td>316</td>
      <td>To Kill A Mockingbird</td>
      <td>Harper Lee</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>1.23</td>
      <td>2019</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>6</th>
      <td>161</td>
      <td>Looking For Alaska</td>
      <td>John Green</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>0.35</td>
      <td>2014</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>7</th>
      <td>222</td>
      <td>The Art Of Racing In The Rain: A Novel</td>
      <td>Garth Stein</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>0.25</td>
      <td>2010</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>8</th>
      <td>223</td>
      <td>The Ballad Of Songbirds And Snakes (A Hunger G...</td>
      <td>Suzanne Collins</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>3.76</td>
      <td>2020</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>9</th>
      <td>229</td>
      <td>The Book Thief</td>
      <td>Markus Zusak</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>0.35</td>
      <td>2014</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>10</th>
      <td>248</td>
      <td>The Girl With The Dragon Tattoo (Millennium Se...</td>
      <td>Stieg Larsson</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>7.99</td>
      <td>2010</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>11</th>
      <td>253</td>
      <td>The Handmaid'S Tale</td>
      <td>Margaret Atwood</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>0.95</td>
      <td>2017</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>12</th>
      <td>315</td>
      <td>Tina Fey: Bossypants</td>
      <td>Tina Fey</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>11.30</td>
      <td>2011</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>13</th>
      <td>33</td>
      <td>Between The World And Me</td>
      <td>Ta-Nehisi Coates</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>7.99</td>
      <td>2015</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>14</th>
      <td>13</td>
      <td>A Wrinkle In Time (Time Quintet)</td>
      <td>Madeleine L'Engle</td>
      <td>4.6</td>
      <td>1697194</td>
      <td>5.35</td>
      <td>2018</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>15</th>
      <td>31</td>
      <td>Becoming</td>
      <td>Michelle Obama</td>
      <td>4.8</td>
      <td>114201</td>
      <td>1.56</td>
      <td>2020</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>16</th>
      <td>11</td>
      <td>A Promised Land</td>
      <td>Barack Obama</td>
      <td>4.9</td>
      <td>110527</td>
      <td>5.44</td>
      <td>2020</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>17</th>
      <td>317</td>
      <td>Too Much And Never Enough: How My Family Creat...</td>
      <td>Mary L. Trump Ph.D.</td>
      <td>4.6</td>
      <td>99089</td>
      <td>10.34</td>
      <td>2020</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>18</th>
      <td>247</td>
      <td>The Girl On The Train</td>
      <td>Paula Hawkins</td>
      <td>4.1</td>
      <td>86655</td>
      <td>12.09</td>
      <td>2015</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>19</th>
      <td>231</td>
      <td>The Boy, The Mole, The Fox And The Horse</td>
      <td>Charlie Mackesy</td>
      <td>4.9</td>
      <td>79923</td>
      <td>13.80</td>
      <td>2020</td>
      <td>Non-Fiction</td>
    </tr>
  </tbody>
</table>
</div>



Looking at the top 20 books with the highest number of reviews, we observe that the top 15 books have the same number of reviews at approximately 1.7 million reviews. The next highest number of reviews is approximately 0.1 million reviews. Upon further inspection on the product pages on Amazon we can see that for the top 15 books are part of a group of products with a shared ratings/reviews section resulting in the significantly higher number of reviews. Separation of the ratings/reviews section to their respective products could not be achieved hence these books will be removed for analysis henceforth.


```python
# Which book has the highest number of reviews?
number_of_reviews = df_no_dup.loc[df_no_dup.reviews < 1000000]

x = number_of_reviews.sort_values('reviews', ascending = False).head(5)['title']
x = x.replace("Too Much And Never Enough: How My Family Created The World'S Most Dangerous Man", "Too Much And Never Enough")
y = number_of_reviews.sort_values('reviews', ascending = False).head(5)['reviews']
```


```python
sns.barplot(x=x, y=y, palette="YlGn")
plt.title('Top 5 Books By Number Of Reviews')
plt.xticks(rotation=45, horizontalalignment='right')
plt.ylabel('No. of Reviews')
plt.xlabel('Book')
```




    Text(0.5, 0, 'Book')




    
![png](/images/amazon_eda/output_29_1.png)
    


<a id='correlation'></a>
### *d. Are ratings, number of reviews, prices, and genre correlated?*
First, we shall look at some descriptive statistics.


```python
# Pie chart for genre
number_genre = df.groupby('fiction flag')[['title']].count().sort_values('title', ascending = False).reset_index()
plt.pie(number_genre['title'], labels=['Non-Fiction','Fiction'], autopct='%1.1f%%', explode = (0,0.05))
plt.title('Percentage Of Books Per Genre')
```




    Text(0.5, 1.0, 'Percentage Of Books Per Genre')




    
![png](/images/amazon_eda/output_31_1.png)
    



```python
# Box plots for rating, number of reviews, price
number_of_reviews.describe()
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
      <th>rating</th>
      <th>reviews</th>
      <th>price</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
      <td>329.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.644073</td>
      <td>17777.963526</td>
      <td>9.323374</td>
      <td>2015.012158</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.213357</td>
      <td>17954.810581</td>
      <td>10.120740</td>
      <td>3.270317</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.300000</td>
      <td>251.000000</td>
      <td>0.250000</td>
      <td>2010.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.500000</td>
      <td>5793.000000</td>
      <td>1.360000</td>
      <td>2012.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.700000</td>
      <td>12103.000000</td>
      <td>7.460000</td>
      <td>2015.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.800000</td>
      <td>23500.000000</td>
      <td>13.950000</td>
      <td>2018.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.900000</td>
      <td>114201.000000</td>
      <td>81.980000</td>
      <td>2020.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, [ax1,ax2,ax3] = plt.subplots(3,1)
sns.boxplot(data=number_of_reviews, x='rating', ax=ax1, color='#f3fab6')
ax1.set_title('Ratings')
sns.boxplot(data=number_of_reviews, x='reviews', ax=ax2, color='#97d385')
ax2.set_title('Reviews')
sns.boxplot(data=number_of_reviews, x='price', ax=ax3, color='#2c8f4b')
ax3.set_title('Price')
plt.tight_layout()
plt.show()
```


    
![png](/images/amazon_eda/output_33_0.png)
    


*Genre:*<br>
1. We observe that there are more Non-Fiction bestsellers than Fiction bestsellers.


For rating, reviews, and price we observe that data is not distributed normally<br>

*Rating:*<br>
1. Small number of outliers with ratings below the 25 percentile of 4.1 rating.

*Reviews:*<br>
1. Data spans a wide range.
2. Small number of outliers with ratings significantly above the 75 percentile of 50k.

*Price:*<br>
1. Small number of outliers with prices significantly above the 75 percentile of $33.

Using a pairplot matrix, we can see if there is any correlation between these 4 variables.


```python
number_of_reviews
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
      <th>title</th>
      <th>author</th>
      <th>rating</th>
      <th>reviews</th>
      <th>price</th>
      <th>year</th>
      <th>fiction flag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10-Day Green Smoothie Cleanse</td>
      <td>JJ Smith</td>
      <td>4.7</td>
      <td>27719</td>
      <td>1.44</td>
      <td>2016</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11/22/63: A Novel</td>
      <td>Stephen King</td>
      <td>4.7</td>
      <td>2588</td>
      <td>2.81</td>
      <td>2011</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12 Rules For Life: An Antidote To Chaos</td>
      <td>Jordan B. Peterson</td>
      <td>4.7</td>
      <td>39960</td>
      <td>7.31</td>
      <td>2018</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1984 (Signet Classics), Book Cover May Vary</td>
      <td>George Orwell</td>
      <td>4.7</td>
      <td>49411</td>
      <td>0.86</td>
      <td>2017</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5,000 Awesome Facts (About Everything!) (Natio...</td>
      <td>National Kids</td>
      <td>4.8</td>
      <td>15160</td>
      <td>2.50</td>
      <td>2019</td>
      <td>Non-Fiction</td>
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
    </tr>
    <tr>
      <th>339</th>
      <td>Winter Of The World: Book Two Of The Century T...</td>
      <td>Ken Follett</td>
      <td>4.6</td>
      <td>14360</td>
      <td>12.97</td>
      <td>2012</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>340</th>
      <td>Women Food And God: An Unexpected Path To Almo...</td>
      <td>Geneen Roth</td>
      <td>4.3</td>
      <td>1721</td>
      <td>15.88</td>
      <td>2010</td>
      <td>Non-Fiction</td>
    </tr>
    <tr>
      <th>341</th>
      <td>Wonder</td>
      <td>R. J. Palacio</td>
      <td>4.8</td>
      <td>30845</td>
      <td>0.35</td>
      <td>2017</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>342</th>
      <td>Wrecking Ball (Diary Of A Wimpy Kid Book 14)</td>
      <td>Jeff Kinney</td>
      <td>4.9</td>
      <td>16016</td>
      <td>1.74</td>
      <td>2019</td>
      <td>Fiction</td>
    </tr>
    <tr>
      <th>343</th>
      <td>You Are A Badass: How To Stop Doubting Your Gr...</td>
      <td>Jen Sincero</td>
      <td>4.7</td>
      <td>28561</td>
      <td>1.17</td>
      <td>2019</td>
      <td>Fiction</td>
    </tr>
  </tbody>
</table>
<p>329 rows × 7 columns</p>
</div>




```python
# Pairplot for correlation
index_vals = number_of_reviews['fiction flag'].astype('category').cat.codes
```


```python
sns.pairplot(number_of_reviews, palette='Set1', hue='fiction flag')
plt.title("Pairplot of Book Statistics")
```




    Text(0.5, 1.0, 'Pairplot of Book Statistics')




    
![png](/images/amazon_eda/output_37_1.png)
    


From the above, we observe no obvious correlation between the variable. However, we can see that the range of ratings between Fiction and Non-Fiction bestsellers differ and shall analyse this difference.

<a id='testing'></a>
### e. *Are the distribution of ratings for Fiction and Non-Fiction books the same?*
Comparison of the distributions will be conducted in two parts:<br>
(i) Testing normality with Shapiro-Wilk test of normality<br>
(ii) Testing statistical differences between the two distributions with Mann-Whitney U test<br>

First, Shapiro-Wilk test will be used to show that ratings are not normally distributed. The bestsellers are than split into two groups, Fiction and Non-Fiction. Mann-Whitney U test is then used to test for statistical differences between the distribution of ratings in these two groups


```python
# (i) Testing normality with Shapiro-Wilk test of normality
alpha = 0.05
stat, pval = shapiro(number_of_reviews['rating'])
print('Statistic:', f'{stat:.3f}')
print('P-value:', f'{pval:.20f}')
if pval > alpha:
    print('Data is distributed normally')
else:
    print('Data is not distributed normally')

# Split bestsellers into two groups
fiction = number_of_reviews[number_of_reviews['fiction flag'] == 'Fiction']['rating']
nonfiction = number_of_reviews[number_of_reviews['fiction flag'] == 'Non-Fiction']['rating']
```

    Statistic: 0.858
    P-value: 0.00000000000000009052
    Data is not distributed normally
    


```python
# (ii) Testing statistical difference between the two distributions with Mann-Whitney U test
stat, pval = mannwhitneyu(nonfiction, fiction)
print('Statistic:', f'{stat:.0f}')
print('P-value:', f'{pval:.5f}')
if pval > alpha:
    print('No significant difference between the two groups')
else:
    print('Significant difference between the two groups')
```

    Statistic: 11205
    P-value: 0.01377
    Significant difference between the two groups
    


```python
sns.displot(data=number_of_reviews, x='rating', hue='fiction flag', palette='icefire', kde=True)
plt.title('Distribution of Ratings between Fiction vs Non-Fiction')
```




    Text(0.5, 1.0, 'Distribution of Ratings between Fiction vs Non-Fiction')




    
![png](/images/amazon_eda/output_41_1.png)
    


Based on the results, we can argue that readers evalute books differently based on the genre, with preferences given to works of Fiction.

***
<a id='conclusion'></a>
## 4. *Conclusion*
In this notebook, our analysis have evaluated the best performing authors and titles in the dataset, and some different perspectives that we can look at the dataset from to obtain additional insights for general questions like 'Which author has the highest average rating'. We also observe that Non-Fiction titles form the majority of bestsellers, but Fiction titles score statistically higher ratings on average suggesting that readers may like works of fiction more.
