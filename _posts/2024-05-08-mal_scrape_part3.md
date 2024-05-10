---
layout: post
title:  "Anime webscraping - Part 3"
date:   2024-05-08 17:57:09 +0800
category: [data_wrangling, misc]
tag: [requests, bs4, html, webscrape, pandas, multiprocessing, fun]
summary: "In this notebook we be scrape a popular anime database/community MyAnimeList, aiming to collect enough raw data from the anime titles available on the website for further processing and learning purposes."
image: /images/banners/mal.png
---

## Contents
1. [Introduction](#1)
2. [Seasonal Title Scraping](#2)
3. [Active Users Scraping](#3)
4. [Conclusion](#4)

<a id='1'></a>
## 1. Introduction
In [previous notebooks](https://wenhao7.github.io/data_wrangling/misc/2024/04/12/mal_scrape_part1.html) we have explored scraping a popular anime database/community [MyAnimeList](https://myanimelist.net) for content and user rating information.  After some [EDA](https://wenhao7.github.io/data_analysis/visualization/misc/2024/04/28/mal_eda.html) and exploration of [Recommendation Systems](https://wenhao7.github.io/data_analysis/machine_learning/visualization/misc/2024/05/03/mal_rec_exploration.html) using the scraped data I have thought of ways to improve our approach to how the data is scraped.

Limitations of the previous implementation that we want to address here:
1. Long run times due to multiple requests required for each Title in serial
2. Sampling a small subset of random active users do not return enough rating data for less popular Titles.

Proposed solutions to the above:
1. Make requests in parallel, reduce number of requests required per Title by 3 orders of magnitude. Achieved by scraping areas of the website listing information for entire seasons of Titles within a single webpage, instead of scraping multiple subpages for each Title that was done prior.
2. Identify Titles that we want more ratings data for, and from the Title's subpage we directly scrape users that have recent interactions with the Title.

In this notebook I will walk through the implementation of the improved approach, the goal being to improve scraping performance and the quality of information that the scripts are returning. This new implementation will also allow us to easily update our anime content/user ratings datasets that are being used by the recommendation system, quickly keeping it up-to-date whenever a new season of titles is released.

<a id='2'></a>
## 2. Seasonal Title Scraping


```python
import os
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import random
import re
import csv
from multiprocessing.pool import ThreadPool as Pool

import logging
import tqdm
```


```python
# Get request from site
site_url = 'https://myanimelist.net'
top_anime_url = site_url + '/anime/season/'
response = requests.get(top_anime_url + '2023/winter')
response.status_code
```




    200



The above webpage contains summary information of all the anime titles that were released during the season, allowing us to scrape relevant data from a large number of titles with a single request, in constrast sending at least one request for each title.

![png](/images/mal/seasonalanime1.png)


```python
# Extract html information from the webpage
doc = BeautifulSoup(response.text)

# Extract relevant portion of the webpage
#row_contents = doc.find_all('div', {'class':'js-anime-category-producer'})
type_contents = doc.find_all('div', {'class':'seasonal-anime-list'})

len(type_contents)
```




    6




```python
total = 0
for i in range(len(type_contents)):
    print('Media Type :', type_contents[i].find('div', {'class':'anime-header'}).text)
    print('Number of Titles: ', len(type_contents[i].find_all('div', {'class':'js-anime-category-producer'})))
    total += len(type_contents[i].find_all('div',{'class':'js-anime-category-producer'}))
print(f'Total Number of Titles This Season: {total}')
          
```

    Media Type : TV (New)
    Number of Titles:  62
    Media Type : TV (Continuing)
    Number of Titles:  64
    Media Type : ONA
    Number of Titles:  83
    Media Type : OVA
    Number of Titles:  11
    Media Type : Movie
    Number of Titles:  25
    Media Type : Special
    Number of Titles:  11
    Total Number of Titles This Season: 256
    

In the above webpage for 2023 Winter season there are 6 different media types with a total of 256 titles in the single response we received.


```python
row_contents = type_contents[0].find_all('div', {'class':'js-anime-category-producer'})
```

```python
# Starting Date, Number of Episodes, Episode Duration
[x.replace(' ','') for x in row_contents[0].find('div', {'class':'prodsrc'}).text.split('\n') if x.replace(' ','') != '']
```




    ['Jan10,2023', '24eps,', '25min']




```python
# Genre
[x.text.strip() for x in row_contents[0].findAll('span', {'class': 'genre'})]
```




    ['Action', 'Adventure', 'Drama']




```python
# Process Number of Episodes and Episode Duration into their own dictionary
def process_prodsrc(row_content):
    content = [x.replace(' ','') for x in row_content.find('div', {'class':'prodsrc'}).text.split('\n') if x.replace(' ','') != '']
    content_dict = {'Episodes':0,'Duration':0}
    for c in content:
        if ('ep' in c or 'eps' in c) and 'Sep' not in c:
            content_dict['Episodes'] = c.replace('eps', '').replace('ep','').replace(',','')
        elif 'min' in c:
            content_dict['Duration'] = c.replace('min', '')
    return content_dict
```


```python
# Studio, Source, Themes, Demographic
[x for x in row_contents[0].find('div', {'class':'properties'}).text.split('\n') if x != '']
```




    ['StudioMAPPA', 'SourceManga', 'ThemesGoreHistorical', 'DemographicSeinen']




```python
# Process Studio, Source, Themes, Demographic into their own dictionary
def process_properties(row_content):
    content = [x for x in row_content.find('div', {'class':'properties'}).text.split('\n') if x != '']
    content_dict = {'Studio':'', 'Source': '', 'Theme': '', 'Demographic': ''}
    for c in content:
        for k in content_dict.keys():
            if k in c:
                content_dict[k] = c.replace(k, '')
    return content_dict
```


```python
# Clean Synopsis text
def clean_text(text):
    text = text.replace('\t','').replace('\n',' ').replace('\r',' ')
    text = re.sub(' +', ' ',text).rstrip('\\').strip()
    return text
```


```python
# Cleaned Synopsis
clean_text(row_contents[0].find('div', {'class':'synopsis'}).text)
```




    "After his father's death and the destruction of his village at the hands of English raiders, Einar wishes for a peaceful life with his family on their newly rebuilt farms. However, fate has other plans: his village is invaded once again. Einar watches helplessly as the marauding Danes burn his lands and slaughter his family. The invaders capture Einar and take him back to Denmark as a slave. Einar clings to his mother's final words to survive. He is purchased by Ketil, a kind slave owner and landlord who promises that Einar can regain his freedom in return for working in the fields. Soon, Einar encounters his new partner in farm cultivation—Thorfinn, a dejected and melancholic slave. As Einar and Thorfinn work together toward their freedom, they are haunted by both sins of the past and the ploys of the present. Yet they carry on, grasping for a glimmer of hope, redemption, and peace in a world that is nothing but unjust and unforgiving. [Written by MAL Rewrite] StudioMAPPA SourceManga ThemesGoreHistorical DemographicSeinen"




```python
# Function to create a dictionary containing all the above information
def extract_info(row_contents, mediatype=''):
    seasonal_contents = []
    for i in range(len(row_contents)):
        prodsrc = process_prodsrc(row_contents[i])
        properties = process_properties(row_contents[i])
        id_ = row_contents[i].find('div', {'class':'genres'})
        title_ = row_contents[i].find('span', {'class':'js-title'})
        score_ = row_contents[i].find('span', {'class':'js-score'})
        members_ = row_contents[i].find('span', {'class':'js-members'})
        start_date_ = row_contents[i].find('span', {'class':'js-start_date'})
        image_ = row_contents[i].find('img')
        
        contents = {
            'MAL_Id': id_.get('id', -1) if id_ else '',
            'Title': title_.text if title_ else '',
            'Image': image_.get('src','') or image_.get('data-src','') if image_ else '',
            'Score': score_.text if score_ else '',
            'Members': members_.text if members_ else '',
            'Start_Date': start_date_.text if start_date_ else '',
            'Episodes': prodsrc['Episodes'],
            'Duration': prodsrc['Duration'],
            'Genres': [x.text.strip() for x in row_contents[i].findAll('span', {'class': 'genre'})],
            'Studio': properties['Studio'],
            'Source': properties['Source'],
            'Themes': re.findall('[A-Z][^A-Z]*', properties['Theme']),
            'Demographic': re.findall('[A-Z][^A-Z]*', properties['Demographic']),
            'Synopsis': clean_text(row_contents[i].find('div', {'class':'synopsis'}).text),
            'Type': mediatype
        }
        seasonal_contents.append(contents)
    return seasonal_contents
```


```python
seasonal_anime = extract_info(row_contents, 'TV')
a = pd.DataFrame(seasonal_anime)
a.head()
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
      <th>Title</th>
      <th>Image</th>
      <th>Score</th>
      <th>Members</th>
      <th>Start_Date</th>
      <th>Episodes</th>
      <th>Duration</th>
      <th>Genres</th>
      <th>Studio</th>
      <th>Source</th>
      <th>Themes</th>
      <th>Demographic</th>
      <th>Synopsis</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49387</td>
      <td>Vinland Saga Season 2</td>
      <td>https://cdn.myanimelist.net/images/anime/1170/...</td>
      <td>8.81</td>
      <td>608171</td>
      <td>20230110</td>
      <td>24</td>
      <td>25</td>
      <td>[Action, Adventure, Drama]</td>
      <td>MAPPA</td>
      <td>Manga</td>
      <td>[Gore, Historical]</td>
      <td>[Seinen]</td>
      <td>After his father's death and the destruction o...</td>
      <td>TV</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52305</td>
      <td>Tomo-chan wa Onnanoko!</td>
      <td>https://cdn.myanimelist.net/images/anime/1444/...</td>
      <td>7.79</td>
      <td>392520</td>
      <td>20230105</td>
      <td>13</td>
      <td>23</td>
      <td>[Comedy, Romance]</td>
      <td>Lay-duce</td>
      <td>Web manga</td>
      <td>[School]</td>
      <td>[]</td>
      <td>Childhood friends Tomo Aizawa and Junichirou "...</td>
      <td>TV</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50608</td>
      <td>Tokyo Revengers: Seiya Kessen-hen</td>
      <td>https://cdn.myanimelist.net/images/anime/1773/...</td>
      <td>7.67</td>
      <td>352516</td>
      <td>20230108</td>
      <td>13</td>
      <td>23</td>
      <td>[Action, Drama, Supernatural]</td>
      <td>LIDENFILMS</td>
      <td>Manga</td>
      <td>[Delinquents, Time , Travel]</td>
      <td>[Shounen]</td>
      <td>In spite of his best time-leaping efforts, Tak...</td>
      <td>TV</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48417</td>
      <td>Maou Gakuin no Futekigousha II: Shijou Saikyou...</td>
      <td>https://cdn.myanimelist.net/images/anime/1369/...</td>
      <td>6.90</td>
      <td>336011</td>
      <td>20230108</td>
      <td>12</td>
      <td>23</td>
      <td>[Action, Fantasy]</td>
      <td>SILVER LINK.</td>
      <td>Light novel</td>
      <td>[Mythology, Reincarnation, School]</td>
      <td>[]</td>
      <td>As peace returns to the demon realm, Anos Vold...</td>
      <td>TV</td>
    </tr>
    <tr>
      <th>4</th>
      <td>50739</td>
      <td>Otonari no Tenshi-sama ni Itsunomanika Dame Ni...</td>
      <td>https://cdn.myanimelist.net/images/anime/1240/...</td>
      <td>7.82</td>
      <td>309405</td>
      <td>20230107</td>
      <td>12</td>
      <td>23</td>
      <td>[Romance]</td>
      <td>Project No.9</td>
      <td>Light novel</td>
      <td>[School]</td>
      <td>[]</td>
      <td>Mahiru Shiina is worthy of her nickname "Angel...</td>
      <td>TV</td>
    </tr>
  </tbody>
</table>
</div>



Our function appears to be working, collecting the relevant information into a dictionary that can be easily converted into a pandas DataFrame.

Next we will include a few useful functions that we will use when scraping all the Titles available on the website, including some logging functionalities to help us track and debug any issues that may occur during the process.


```python
# Helper functions
### Implement randomized sleep time in between requests to reduce chance of being blocked from site
def sleep(t=3):
    rand_t = random.random() * (t) +0.5
    time.sleep(rand_t)
    
### Save our dictionary to a .csv file
def write_seasonal_csv(items, path):
    written_id = set()
    
    # Assign header names with handling of seasons with no new release in certain media types
    for i in range(len(items)):
        if items[i]:
            headers = list(items[i][0].keys())
            break
    
    # In case no new titles released
    if headers:
        # Open the file in write mode
        if not path in os.listdir():
            with open(path, 'w', encoding='utf-8') as f:
                # Return if there's nothing to write
                if len(items) == 0:
                    return
                # Write the headers in the first line
                f.write('|'.join(headers) + '\n')

        with open(path, 'a', encoding='utf-8') as f:
            # Write one item per line
            for i in range(len(items)):
                for item in items[i]:
                    values = []
                    # Check if title has already been added to prevent duplicated entries, some shows span multiple seasons
                    if item.get('MAL_Id') in written_id:
                        continue
                    for header in headers:
                        values.append(str(item.get(header, "")).replace('|',''))
                    f.write('|'.join(values) + "\n")          
                    written_id.add(item.get('Id'))

### Send request to website
def get_response(url):    
    # Try for up to 3 times per URL
    for _ in range(3):
        try:
            sleep(3)
            response = requests.get(url, headers=req_head)
            # If response is good we return the BS object for further processing
            if response.status_code == 200:
                doc = BeautifulSoup(response.text)
                row_contents = doc.find_all('div', {'class':'seasonal-anime-list'})
                if row_contents is None:
                    logging.warning(f'row_contents is None for {url}')
                    print(f'----------- row_contents is None for {url} ------------')
                return row_contents
            
            # If response suggests we are rate limited, make this thread sleep for ~3 minutes before continuing on next loop
            elif response.status_code == 429 or response.status_code == 405:
                logging.warning(f'{response.status_code} for {url}')
                print(f'----------- {response.status_code} occured for {url} ------------')
                buffer_t = random.random() * (40) + 160
                sleep(buffer_t)
                continue
            
            # Any other unexpected response
            else:
                logging.warning(f'{response.status_code} for {url}')
                print(f'----------- {response.status_code} occured for {url} ------------')
                sleep(5)
                continue
        
        # Any unexpected issues with sending request
        except:
            logging.error('Error trying to send request')
            buffer_t = random.random() * (40) + 100
            sleep(buffer_t)
            continue            
    print("-----------------------------Error sending request-----------------------------")
    print(time.asctime())

    
# Instantiate variables
start_year, end_year = 1917, 2024
seasons = ['winter','spring','summer','fall']
req_head = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
           'X-MAL-CLIENT-ID':'e09c24c7eb88c3f399d9bd1355b4e015'}
seasonal_anime_filename = 'seasonal_anime.csv'

logging.basicConfig(filename='seasonal.log', filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
```


```python
# Scrape all URLs between start and end years for specified seasons, 
# multiprocessing available, option to override with a specified list of url available as well
def scrape(file_name, start_year=1917, end_year=2024, seasons=['winter','spring','summer','fall'], req=req_head, nprocesses=4, url_list=None):
    top_anime_url = 'https://myanimelist.net/anime/season/'
    
    # If specific URLs are not provided, a list of URLs will be generated based on start/end years and seasons provided.
    if not url_list:
        url_list = [top_anime_url + str(year) + '/' + str(season) for year in range(start_year,end_year+1) for season in seasons]
    
    anime_list = []
    # nprocesses number of threads processing URL list in sequence parallelly
    with Pool(processes=nprocesses) as pool:
        for type_contents in tqdm.tqdm(pool.imap(get_response, url_list), total=len(url_list)):
            if type_contents is None:
                continue
            for i in range(len(type_contents)):
                row_contents = type_contents[i].find_all('div', {'class':'js-anime-category-producer'})
                mediatype = type_contents[i].find('div', {'class':'anime-header'}).text
                seasonal_contents = extract_info(row_contents, mediatype)
                anime_list.append(seasonal_contents)
            sleep(5) # a few seconds sleep before next request is sent to avoid rate limit by site
    
    # Write scraped data to disk
    write_seasonal_csv(anime_list, file_name)
    return anime_list
```


```python
a = scrape(seasonal_anime_filename, start_year = 1917, end_year = 2024)
```

     21%|████████████████▋                                                                | 89/432 [04:48<17:23,  3.04s/it]

    ----------- 405 occured for https://myanimelist.net/anime/season/2000/fall ------------
    ----------- 405 occured for https://myanimelist.net/anime/season/2001/spring ------------
    ----------- 405 occured for https://myanimelist.net/anime/season/2001/winter ------------
    

     21%|████████████████▉                                                                | 90/432 [04:52<19:47,  3.47s/it]

    ----------- 405 occured for https://myanimelist.net/anime/season/2001/summer ------------
    

     22%|█████████████████▋                                                               | 94/432 [05:04<15:54,  2.83s/it]

    ----------- 405 occured for https://myanimelist.net/anime/season/2001/spring ------------
    

    100%|████████████████████████████████████████████████████████████████████████████████| 432/432 [22:52<00:00,  3.18s/it]
    


```python
#write_seasonal_csv(a,seasonal_anime_filename)
```




The above operation scraped all seasonal pages available on the website in just 20 minutes, a significant improvement from the previous implementation which took multiple hours just to scrape a subset of Titles on the website.

In the output we see that we appear to have been rate limited once within the 20 minutes. As the implemented logic will retry the same request thrice and there are no errors that repeated for three times, I assume that all requests were successful. But as a sanity check we will investigate these errors.


```python
# Collect logged failed urls
with open('seasonal.log') as f:
    f = f.readlines()
failed_urls = []
pattern = r'https://myanimelist.net/anime/season/[ a-zA-Z0-9./]+/[ a-zA-Z0-9./]+'
for line in f:
    url = re.findall(pattern, line)
    if url:
        failed_urls.append(url[0])
len(failed_urls)
```




    5




```python
# Remove duplicated urls
failed_urls = list(dict.fromkeys(failed_urls))
len(failed_urls)
```




    4



4 unique URLs faced rate limiting errors during scraping, these are urls for 2001 and 2002 seasons


```python
df = pd.read_csv('seasonal_anime.csv')
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
      <th>MAL_Id</th>
      <th>Title</th>
      <th>Image</th>
      <th>Score</th>
      <th>Members</th>
      <th>Start_Date</th>
      <th>Episodes</th>
      <th>Duration</th>
      <th>Genres</th>
      <th>Studio</th>
      <th>Source</th>
      <th>Themes</th>
      <th>Demographic</th>
      <th>Synopsis</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23189</td>
      <td>Dekobou Shingachou: Meian no Shippai</td>
      <td>https://cdn.myanimelist.net/images/qm_50.gif</td>
      <td>5.84</td>
      <td>1544</td>
      <td>19170200</td>
      <td>1</td>
      <td>5</td>
      <td>['Comedy']</td>
      <td>Unknown</td>
      <td>Original</td>
      <td>[]</td>
      <td>[]</td>
      <td>A man first realizes he's born to be a samurai...</td>
      <td>Movie</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17387</td>
      <td>Imokawa Mukuzo: Genkanban no Maki</td>
      <td>https://cdn.myanimelist.net/images/qm_50.gif</td>
      <td>5.26</td>
      <td>1133</td>
      <td>19170100</td>
      <td>1</td>
      <td>8</td>
      <td>['Comedy']</td>
      <td>Unknown</td>
      <td>Original</td>
      <td>[]</td>
      <td>[]</td>
      <td>The third professionally produced Japanese ani...</td>
      <td>Movie</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6654</td>
      <td>Namakura Gatana</td>
      <td>https://cdn.myanimelist.net/images/anime/1959/...</td>
      <td>5.50</td>
      <td>9633</td>
      <td>19170630</td>
      <td>1</td>
      <td>4</td>
      <td>['Comedy']</td>
      <td>Unknown</td>
      <td>Original</td>
      <td>['Samurai']</td>
      <td>[]</td>
      <td>Namakura Gatana  meaning "dull-edged sword " i...</td>
      <td>Movie</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10742</td>
      <td>Saru to Kani no Gassen</td>
      <td>https://cdn.myanimelist.net/images/anime/4/837...</td>
      <td>4.93</td>
      <td>1146</td>
      <td>19170520</td>
      <td>1</td>
      <td>6</td>
      <td>['Drama']</td>
      <td>Unknown</td>
      <td>Other</td>
      <td>[]</td>
      <td>[]</td>
      <td>A monkey tricks a crab and steals his food. Mo...</td>
      <td>Movie</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24575</td>
      <td>Yume no Jidousha</td>
      <td>https://cdn.myanimelist.net/images/qm_50.gif</td>
      <td>5.62</td>
      <td>623</td>
      <td>19170500</td>
      <td>1</td>
      <td>0</td>
      <td>[]</td>
      <td>Unknown</td>
      <td>Original</td>
      <td>['Racing']</td>
      <td>[]</td>
      <td>It is most likely a story about a great dream ...</td>
      <td>Movie</td>
    </tr>
  </tbody>
</table>
</div>




```python
# impute missing days/months in start_date
def impute_day(date):
    if str(date)[-2:] == '00':
        date = str(date)[:-2] + '01'
    if str(date)[4:-2] == '00':
        date = str(date)[:4] + '01' + str(date)[-2:]
    return date
```


```python
df.Start_Date = df.Start_Date.apply(impute_day)
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
      <th>MAL_Id</th>
      <th>Title</th>
      <th>Image</th>
      <th>Score</th>
      <th>Members</th>
      <th>Start_Date</th>
      <th>Episodes</th>
      <th>Duration</th>
      <th>Genres</th>
      <th>Studio</th>
      <th>Source</th>
      <th>Themes</th>
      <th>Demographic</th>
      <th>Synopsis</th>
      <th>Type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23189</td>
      <td>Dekobou Shingachou: Meian no Shippai</td>
      <td>https://cdn.myanimelist.net/images/qm_50.gif</td>
      <td>5.84</td>
      <td>1544</td>
      <td>19170201</td>
      <td>1</td>
      <td>5</td>
      <td>['Comedy']</td>
      <td>Unknown</td>
      <td>Original</td>
      <td>[]</td>
      <td>[]</td>
      <td>A man first realizes he's born to be a samurai...</td>
      <td>Movie</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17387</td>
      <td>Imokawa Mukuzo: Genkanban no Maki</td>
      <td>https://cdn.myanimelist.net/images/qm_50.gif</td>
      <td>5.26</td>
      <td>1133</td>
      <td>19170101</td>
      <td>1</td>
      <td>8</td>
      <td>['Comedy']</td>
      <td>Unknown</td>
      <td>Original</td>
      <td>[]</td>
      <td>[]</td>
      <td>The third professionally produced Japanese ani...</td>
      <td>Movie</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6654</td>
      <td>Namakura Gatana</td>
      <td>https://cdn.myanimelist.net/images/anime/1959/...</td>
      <td>5.50</td>
      <td>9633</td>
      <td>19170630</td>
      <td>1</td>
      <td>4</td>
      <td>['Comedy']</td>
      <td>Unknown</td>
      <td>Original</td>
      <td>['Samurai']</td>
      <td>[]</td>
      <td>Namakura Gatana  meaning "dull-edged sword " i...</td>
      <td>Movie</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10742</td>
      <td>Saru to Kani no Gassen</td>
      <td>https://cdn.myanimelist.net/images/anime/4/837...</td>
      <td>4.93</td>
      <td>1146</td>
      <td>19170520</td>
      <td>1</td>
      <td>6</td>
      <td>['Drama']</td>
      <td>Unknown</td>
      <td>Other</td>
      <td>[]</td>
      <td>[]</td>
      <td>A monkey tricks a crab and steals his food. Mo...</td>
      <td>Movie</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24575</td>
      <td>Yume no Jidousha</td>
      <td>https://cdn.myanimelist.net/images/qm_50.gif</td>
      <td>5.62</td>
      <td>623</td>
      <td>19170501</td>
      <td>1</td>
      <td>0</td>
      <td>[]</td>
      <td>Unknown</td>
      <td>Original</td>
      <td>['Racing']</td>
      <td>[]</td>
      <td>It is most likely a story about a great dream ...</td>
      <td>Movie</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.Start_Date = pd.to_datetime(df.Start_Date, format='%Y%m%d')
df.shape
```




    (28744, 15)



The Start_Date in the above dataframe has been imputed and converted to datetime format, now we can conduct our sanity checks on the years that produced the errors while scraping.


```python
df[df.Start_Date.dt.year == 2001]['Start_Date'].dt.month.value_counts()
```




    Start_Date
    10    112
    4      85
    7      56
    12     43
    8      39
    3      34
    5      32
    1      29
    6      22
    11     22
    2      21
    9      21
    Name: count, dtype: int64




```python
df[df.Start_Date.dt.year == 2002]['Start_Date'].dt.month.value_counts()
```




    Start_Date
    4     121
    10     87
    1      52
    8      36
    11     36
    3      34
    7      34
    12     33
    5      26
    9      24
    2      22
    6      20
    Name: count, dtype: int64



We see that both 2001 and 2002 have many number of titles released throughout each of the twelve months of the year, suggesting that all four seasons we successfully scraped for both years.

<a id='3'></a>
## 3. Active Users Scraping
Now we are going to explore selectively scraping active users that have rated a Title that we want to collect more ratings for. Together with the seasonal scraping from the previous section this will allow the option of quickly updating the capabilities of the Recommendation System to include the newest titles into consideration.

The below image shows a snippet of what this section looks like on the webpage.
![png](/images/mal/recentmembers.png)


```python
response = requests.get('https://myanimelist.net/anime/49458/Kono_Subarashii_Sekai_ni_Shukufuku_wo_3/stats?show=0#members',headers=req_head)
doc = BeautifulSoup(response.text)
row_contents = doc.find_all('table', {'class':'table-recently-updated'})

# We expect to see Username , Score, Status, Eps Seen, Activity
[x.text for x in row_contents[0].findAll('tr')[1].findAll('td')]
```




    ['dawid550', '-', 'Plan to Watch', '', '36 minutes ago']




```python
# Loop through found users and collect users that have rated the Title
res_list = []
for i in range(len(row_contents[0].findAll('tr'))):
    res = [x.text for x in row_contents[0].findAll('tr')[i].findAll('td')]
    if res[1] != '-':
        res_list.append(res)
pd.DataFrame(res_list[1:], columns=res_list[0])
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
      <th>Member</th>
      <th>Score</th>
      <th>Status</th>
      <th>Eps Seen</th>
      <th>Activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fxl2</td>
      <td>6</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>37 minutes ago</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Djimbe</td>
      <td>10</td>
      <td>Watching</td>
      <td>\n                        3 / 11\n            ...</td>
      <td>38 minutes ago</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Prettig</td>
      <td>10</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>38 minutes ago</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CzechAnime</td>
      <td>10</td>
      <td>Watching</td>
      <td>\n                        5 / 11\n            ...</td>
      <td>42 minutes ago</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fajar38</td>
      <td>7</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>42 minutes ago</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Naitchu</td>
      <td>9</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>43 minutes ago</td>
    </tr>
    <tr>
      <th>6</th>
      <td>MishMashMoshi</td>
      <td>7</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>44 minutes ago</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GriffonLord</td>
      <td>8</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>45 minutes ago</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Danderfluff</td>
      <td>7</td>
      <td>Watching</td>
      <td>\n                        5 / 11\n            ...</td>
      <td>46 minutes ago</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MrJast</td>
      <td>10</td>
      <td>Watching</td>
      <td>\n                        11 / 11\n           ...</td>
      <td>46 minutes ago</td>
    </tr>
    <tr>
      <th>10</th>
      <td>kawaiigabz</td>
      <td>9</td>
      <td>Watching</td>
      <td>\n                        3 / 11\n            ...</td>
      <td>48 minutes ago</td>
    </tr>
    <tr>
      <th>11</th>
      <td>boknight</td>
      <td>6</td>
      <td>Watching</td>
      <td>\n                        1 / 11\n            ...</td>
      <td>49 minutes ago</td>
    </tr>
    <tr>
      <th>12</th>
      <td>fabian332</td>
      <td>7</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>50 minutes ago</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mkody</td>
      <td>8</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>50 minutes ago</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Divyansenpai69</td>
      <td>8</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>50 minutes ago</td>
    </tr>
    <tr>
      <th>15</th>
      <td>human4ever</td>
      <td>7</td>
      <td>Watching</td>
      <td>\n                        2 / 11\n            ...</td>
      <td>55 minutes ago</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Travaughn13</td>
      <td>10</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>55 minutes ago</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dnilos911</td>
      <td>9</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>58 minutes ago</td>
    </tr>
    <tr>
      <th>18</th>
      <td>LouLouLouLouLou</td>
      <td>10</td>
      <td>Watching</td>
      <td>\n                        - / 11\n            ...</td>
      <td>59 minutes ago</td>
    </tr>
    <tr>
      <th>19</th>
      <td>sorotomi97</td>
      <td>8</td>
      <td>Watching</td>
      <td>\n                        3 / 11\n            ...</td>
      <td>1 hour ago</td>
    </tr>
    <tr>
      <th>20</th>
      <td>mrbacon56</td>
      <td>10</td>
      <td>Watching</td>
      <td>\n                        4 / 11\n            ...</td>
      <td>1 hour ago</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Plugma</td>
      <td>8</td>
      <td>Completed</td>
      <td>\n                        11 / 11\n           ...</td>
      <td>1 hour ago</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Kitto999</td>
      <td>9</td>
      <td>Watching</td>
      <td>\n                        2 / 11\n            ...</td>
      <td>1 hour ago</td>
    </tr>
    <tr>
      <th>23</th>
      <td>KeerthiVasanG</td>
      <td>8</td>
      <td>Watching</td>
      <td>\n                        5 / 11\n            ...</td>
      <td>1 hour ago</td>
    </tr>
    <tr>
      <th>24</th>
      <td>LeviAck25</td>
      <td>10</td>
      <td>Watching</td>
      <td>\n                        1 / 11\n            ...</td>
      <td>1 hour ago</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Koukyy</td>
      <td>8</td>
      <td>Watching</td>
      <td>\n                        2 / 11\n            ...</td>
      <td>1 hour ago</td>
    </tr>
  </tbody>
</table>
</div>



We see that out of 75 users in the response we have 25 users who have rated the title! From here we can reuse our previous user scraping script to obtain these users' rating lists through the official API of the website so our system can use that information for collaborative filtering.

Should we require additional number of user ratings we can increment the number of members in the url to scrape additional pages of recent interactions. By design the webpage will return pages of 75 members and we have scraped the page starting at interaction number 0 to 74. In our scripts we can increment it by 75 each time until the desired number of usernames are obtained.

Next we can quickly go through the possible Titles that we need to obtain additional user ratings information from.


```python
user_ratings = pd.read_csv('cleaned_user_ratings.csv')
user_ratings[user_ratings.Rating_Score != 0].Anime_Id.unique()
print(f'Number of unique titles scraped : {len(df)}')
print(f'Number of unique titles rated in our user dataset : {user_ratings[user_ratings.Rating_Score != 0].Anime_Id.nunique()}')
```

    Number of unique titles scraped : 18062
    Number of unique titles rated in our user dataset : 14914
    


```python
user_ratings[~user_ratings['Anime_Id'].isin(df.Id)].sort_values('Rating_Score', ascending=False)
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
      <th>4946590</th>
      <td>matti_god</td>
      <td>18193</td>
      <td>57435</td>
      <td>Street Fighter 6 x Spy x Family Movie: Code: W...</td>
      <td>completed</td>
      <td>10</td>
      <td>1</td>
      <td>False</td>
      <td>2023-12-19 15:00:22+00:00</td>
      <td>2023-12-04</td>
    </tr>
    <tr>
      <th>3356659</th>
      <td>ShiroAlex</td>
      <td>12378</td>
      <td>52745</td>
      <td>Liella no Uta 2</td>
      <td>completed</td>
      <td>10</td>
      <td>12</td>
      <td>False</td>
      <td>2024-04-03 10:35:45+00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>531240</th>
      <td>potatoxslayer</td>
      <td>1938</td>
      <td>37954</td>
      <td>Neo-Aspect</td>
      <td>completed</td>
      <td>10</td>
      <td>1</td>
      <td>False</td>
      <td>2019-01-28 05:02:56+00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>535345</th>
      <td>45rfew</td>
      <td>1953</td>
      <td>32807</td>
      <td>Xiong Chumo</td>
      <td>completed</td>
      <td>10</td>
      <td>104</td>
      <td>False</td>
      <td>2022-08-01 03:22:35+00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>535348</th>
      <td>45rfew</td>
      <td>1953</td>
      <td>32818</td>
      <td>Xiong Chumo: Huanqiu Da Maoxian</td>
      <td>completed</td>
      <td>10</td>
      <td>104</td>
      <td>False</td>
      <td>2022-08-01 03:22:44+00:00</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>2143827</th>
      <td>SwopaKing</td>
      <td>7873</td>
      <td>56906</td>
      <td>Isekai de Cheat Skill wo Te ni Shita Ore wa, G...</td>
      <td>plan_to_watch</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>2023-10-15 12:33:58+00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2143639</th>
      <td>M3m3supreme</td>
      <td>7872</td>
      <td>49233</td>
      <td>Youjo Senki II</td>
      <td>plan_to_watch</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>2021-06-19 16:34:22+00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2143619</th>
      <td>M3m3supreme</td>
      <td>7872</td>
      <td>34453</td>
      <td>Uma Musume: Pretty Derby PV</td>
      <td>plan_to_watch</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>2018-04-03 15:47:56+00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2143567</th>
      <td>M3m3supreme</td>
      <td>7872</td>
      <td>53065</td>
      <td>Sono Bisque Doll wa Koi wo Suru (Zoku-hen)</td>
      <td>plan_to_watch</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>2022-09-18 10:22:21+00:00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5452183</th>
      <td>mintcakee</td>
      <td>20010</td>
      <td>23057</td>
      <td>Yukidoke</td>
      <td>plan_to_watch</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>2023-10-23 10:12:55+00:00</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>93341 rows × 10 columns</p>
</div>




```python
print(f"Number of titles in user rating data that does not appear in scraped seasonal data : {user_ratings[~user_ratings['Anime_Id'].isin(df.Id)].Anime_Id.nunique()}")
```

    Number of titles in user rating data that does not appear in scraped seasonal data : 3036
    

We see approximately 3000 titles appearing in our user ratings dataset, but they do no exist within our scraped aired titles. Upon further investigation it appears that these titles are scheduled to release in the further, or are promotional videos that do not fall under the category of a "proper show" and hence are excluded from the seasonal roster.


```python
print(f'Number of titles within scraped seasonal data missing from user rating data : {len(df[~df.Id.isin(user_ratings[user_ratings.Rating_Score != 0].Anime_Id.unique())])}')
```

    Number of titles within scraped seasonal data missing from user rating data : 5502
    

With more than 5000 titles it means an equivalent number of requests will be required to obtain the recent user interactions information. Extrapolating from the time required to scrape our seasonal data this would mean almost 4 hours to go through all 5000+ titles!

As an alternative we can ignore the really obscure Titles that also have low rating scores under the assumption that watchers would be less likely to enjoy them anyway. The likelihood of these titles getting recommended is also low as they would not rank high during collaborative filtering due to their low scores and low number of ratings.


```python
df[(~df.Id.isin(user_ratings[user_ratings.Rating_Score != 0].Anime_Id.unique()))].describe()
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
      <th>Id</th>
      <th>Score</th>
      <th>Members</th>
      <th>Start_Date</th>
      <th>Duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5502.000000</td>
      <td>5502.000000</td>
      <td>5502.00000</td>
      <td>5502</td>
      <td>5502.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>33578.684842</td>
      <td>2.237177</td>
      <td>1041.49691</td>
      <td>2008-09-22 08:29:50.185387008</td>
      <td>24.303708</td>
    </tr>
    <tr>
      <th>min</th>
      <td>217.000000</td>
      <td>0.000000</td>
      <td>7.00000</td>
      <td>1917-05-01 00:00:00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18629.500000</td>
      <td>0.000000</td>
      <td>168.00000</td>
      <td>2001-11-26 12:00:00</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>36669.500000</td>
      <td>0.000000</td>
      <td>393.00000</td>
      <td>2013-06-09 00:00:00</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48209.750000</td>
      <td>5.670000</td>
      <td>859.50000</td>
      <td>2018-08-17 18:00:00</td>
      <td>29.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>58805.000000</td>
      <td>7.970000</td>
      <td>132342.00000</td>
      <td>2024-12-01 00:00:00</td>
      <td>167.000000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>17495.517912</td>
      <td>2.899072</td>
      <td>3890.49838</td>
      <td>NaN</td>
      <td>27.931342</td>
    </tr>
  </tbody>
</table>
</div>



For demonstration purposes I would use 2000 members and a score of at least 6 as our filter.


```python
df[(~df.Id.isin(user_ratings[user_ratings.Rating_Score != 0].Anime_Id.unique())) & (df.Members > 2000) & (df.Score >= 6)]
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
      <th>Id</th>
      <th>Title</th>
      <th>Image</th>
      <th>Score</th>
      <th>Members</th>
      <th>Start_Date</th>
      <th>Episodes</th>
      <th>Duration</th>
      <th>Genre</th>
      <th>Studio</th>
      <th>Source</th>
      <th>Themes</th>
      <th>Demographic</th>
      <th>Synopsis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>133</th>
      <td>4948</td>
      <td>Shounen Sarutobi Sasuke</td>
      <td>https://cdn.myanimelist.net/images/anime/1266/...</td>
      <td>6.27</td>
      <td>2238</td>
      <td>1959-12-25</td>
      <td>1</td>
      <td>82</td>
      <td>['Adventure'  'Fantasy']</td>
      <td>Toei Animation</td>
      <td>Original</td>
      <td>[]</td>
      <td>[]</td>
      <td>Magic Boy was the first ever Japanese animatio...</td>
    </tr>
    <tr>
      <th>149</th>
      <td>2686</td>
      <td>Tetsujin 28-gou</td>
      <td>https://cdn.myanimelist.net/images/anime/8/717...</td>
      <td>6.94</td>
      <td>3872</td>
      <td>1963-10-20</td>
      <td>96</td>
      <td>25</td>
      <td>['Adventure'  'Sci-Fi']</td>
      <td>Eiken</td>
      <td>Manga</td>
      <td>['Mecha']</td>
      <td>['Shounen']</td>
      <td>Dr.Haneda was developing experimental giant ro...</td>
    </tr>
    <tr>
      <th>209</th>
      <td>3900</td>
      <td>Ougon Bat</td>
      <td>https://cdn.myanimelist.net/images/anime/2/286...</td>
      <td>6.88</td>
      <td>2816</td>
      <td>1967-04-01</td>
      <td>52</td>
      <td>25</td>
      <td>['Action'  'Sci-Fi']</td>
      <td>sDai-Ichi DougaDongyang Animation</td>
      <td>Novel</td>
      <td>['Super '  'Power']</td>
      <td>[]</td>
      <td>A golden warrior wearing a cape and a scepter ...</td>
    </tr>
    <tr>
      <th>229</th>
      <td>5834</td>
      <td>Kyojin no Hoshi</td>
      <td>https://cdn.myanimelist.net/images/anime/12/59...</td>
      <td>7.47</td>
      <td>3206</td>
      <td>1968-03-30</td>
      <td>182</td>
      <td>25</td>
      <td>['Drama'  'Sports']</td>
      <td>TMS Entertainment</td>
      <td>Manga</td>
      <td>['Team '  'Sports']</td>
      <td>['Shounen']</td>
      <td>The story is about Hyuma Hoshi  a promising yo...</td>
    </tr>
    <tr>
      <th>238</th>
      <td>5997</td>
      <td>Sabu to Ichi Torimono Hikae</td>
      <td>https://cdn.myanimelist.net/images/anime/9/840...</td>
      <td>7.04</td>
      <td>2748</td>
      <td>1968-10-03</td>
      <td>52</td>
      <td>25</td>
      <td>['Action'  'Adventure'  'Drama'  'Slice of Life']</td>
      <td>Toei Animation</td>
      <td>Manga</td>
      <td>['Detective'  'Historical'  'Martial '  'Arts'...</td>
      <td>['Shounen']</td>
      <td>The series follows the adventures of Sabu  a y...</td>
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
    </tr>
    <tr>
      <th>17902</th>
      <td>56840</td>
      <td>T.P BON</td>
      <td>https://cdn.myanimelist.net/images/anime/1003/...</td>
      <td>6.95</td>
      <td>3693</td>
      <td>2024-05-02</td>
      <td>?</td>
      <td>28</td>
      <td>['Action'  'Adventure']</td>
      <td>Bones</td>
      <td>Manga</td>
      <td>['Time '  'Travel']</td>
      <td>[]</td>
      <td>An ordinary high school student named Bon beco...</td>
    </tr>
    <tr>
      <th>17904</th>
      <td>58689</td>
      <td>Yuanshen: Jinzhong Ge</td>
      <td>https://cdn.myanimelist.net/images/anime/1229/...</td>
      <td>7.97</td>
      <td>2176</td>
      <td>2024-04-17</td>
      <td>1</td>
      <td>7</td>
      <td>['Action'  'Drama'  'Fantasy']</td>
      <td>Unknown</td>
      <td>Game</td>
      <td>[]</td>
      <td>[]</td>
      <td>Animated short film about the backstory of the...</td>
    </tr>
    <tr>
      <th>17940</th>
      <td>54866</td>
      <td>Blue Lock: Episode Nagi</td>
      <td>https://cdn.myanimelist.net/images/anime/1239/...</td>
      <td>6.86</td>
      <td>36190</td>
      <td>2024-04-19</td>
      <td>1</td>
      <td>91</td>
      <td>['Sports']</td>
      <td>8bit</td>
      <td>Manga</td>
      <td>['Team '  'Sports']</td>
      <td>['Shounen']</td>
      <td>A spin-off series of Blue Lock focusing on Sei...</td>
    </tr>
    <tr>
      <th>17946</th>
      <td>57478</td>
      <td>Kuramerukagari</td>
      <td>https://cdn.myanimelist.net/images/anime/1764/...</td>
      <td>6.33</td>
      <td>5131</td>
      <td>2024-04-12</td>
      <td>1</td>
      <td>61</td>
      <td>['Mystery'  'Sci-Fi'  'Suspense']</td>
      <td>Team OneOne</td>
      <td>Original</td>
      <td>['Detective']</td>
      <td>[]</td>
      <td>This is a story that weaves together people an...</td>
    </tr>
    <tr>
      <th>17949</th>
      <td>56553</td>
      <td>Kurayukaba</td>
      <td>https://cdn.myanimelist.net/images/anime/1885/...</td>
      <td>6.48</td>
      <td>4702</td>
      <td>2024-04-12</td>
      <td>1</td>
      <td>63</td>
      <td>['Mystery'  'Sci-Fi'  'Suspense']</td>
      <td>Team OneOne</td>
      <td>Original</td>
      <td>['Detective']</td>
      <td>[]</td>
      <td>Business is slow for the Ootsuji Detective Age...</td>
    </tr>
  </tbody>
</table>
<p>301 rows × 14 columns</p>
</div>



Instantly we have cut the number of Titles we want to scrape down to 301, which will complete in less than 15 minutes. 

With this, we can expect to collect enough information to add Titles from a new season to our recommendation system within 30 minutes! Approximately 200 requests (1 to obtain seasonal titles + recent user interactions from average of 150 titles released per season) which would take about 10 minutes, and an additional 20 minutes to obtain the user ratings data from the official API which is more lenient with its rate limiting in my experience.

<a id ='4'></a>
## 4. Conclusion
In this notebook we have explore an improved implementation of our previous webscraping approach to obtaining content information and user ratings information from the website. This implementation cuts down the time taken to scrape our content information by over 90%, taking only 20 minutes to scrape the entire site now. 

This targeted user approach improves the quality of user rating data that we will obtain. Previously we scraped user data indiscriminately as long as they are recently active, resulting in a lot of obscure titles missing from the dataset, and a lot of wasted API calls when the active user scraped does not maintain a useful rating list. The updated approach obtains the usernames directly from the Title's recent interactions list, ensuring that the scraped user ratings data will at least contain information of the Title of interest.
