---
layout: post
title:  "Dirty Data Samples - Cleaning Data With Pandas - Part 1"
date:   2024-04-04 19:31:09 +0800
category: [data_wrangling, misc]
tag: [numpy, pandas, fun]
summary: "In this notebook we will explore 4 dirty datasets sourced from the internet that has been structured poorly and clean them using Pandas package in Python."
image: /images/banners/data_cleaning.png
---

## Contents
0. [Introduction](#0)
1. [Badly Structured Sales Data 1](#1)
2. [Badly Structured Sales Data 2](#2)
3. [Badly Structured Sales Data 3](#3)
4. [Badly Structured Sales Data 4](#4)

<a id='0'></a>
## 0. Introduction
In this notebook we will explore 4 "dirty" datasets from a [power bi resources website](https://foresightbi.com.ng/microsoft-power-bi/dirty-data-samples-to-practice-on/) containing data that has been structured poorly. Utilizing pandas we will restructure the dataset so that the datasets are in a format ready for further processing/exploration. The website also shows the final requirements that they want the "clean" dataset to look like, and we will be working with that in mind.

This is part one of two notebooks working with these "dirty" datasets. In [Part 2](/data_wrangling/misc/2024/04/05/data_cleaning_part2.html) we will be looking at another 4 datasets that contains data that have poorly recorded values that requires rectification.


```python
import os
import pandas as pd
import numpy as np
files = [f for f in os.listdir('.') if (os.path.isfile(f) and 'xlsx' in f)]
files
```




    ['1.-Badly-Structured-Sales-Data-1.xlsx',
     '2.-Badly-Structured-Sales-Data-2.xlsx',
     '3.-Badly-Structured-Sales-Data-3.xlsx',
     '4.-Badly-Structured-Sales-Data-4.xlsx',
     '5.-Jumbled-up-Customers-Details.xlsx',
     '6.-Hospital-Data-with-Mixed-Numbers-and-Characters.xlsx',
     '7.-Medicine-Data-with-lumped-Quantity-and-Measure.xlsx',
     '8.-Invoices-with-Merged-Categories-and-Merged-Amounts.xlsx']



<a id='1'></a>
## 1. Badly Strutured Sales Data 1
In this section we will re-arrange the data into a long format with 4 columns, the original data has a mix of rows and columns everywhere almost in a wide foramt (but not quite!).


```python
data1 = pd.read_excel('1.-Badly-Structured-Sales-Data-1.xlsx')
df = data1.copy()
df
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
      <th>Segment&gt;&gt;</th>
      <th>Consumer</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Consumer Total</th>
      <th>Corporate</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Corporate Total</th>
      <th>Home Office</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
      <th>Unnamed: 14</th>
      <th>Home Office Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ship Mode&gt;&gt;</td>
      <td>First Class</td>
      <td>Same Day</td>
      <td>Second Class</td>
      <td>Standard Class</td>
      <td>NaN</td>
      <td>First Class</td>
      <td>Same Day</td>
      <td>Second Class</td>
      <td>Standard Class</td>
      <td>NaN</td>
      <td>First Class</td>
      <td>Same Day</td>
      <td>Second Class</td>
      <td>Standard Class</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order ID</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>CA-2011-100293</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.056</td>
      <td>91.0560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-100706</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.44</td>
      <td>NaN</td>
      <td>129.440</td>
      <td>NaN</td>
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
      <td>CA-2011-100895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>605.47</td>
      <td>605.470</td>
      <td>NaN</td>
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
    </tr>
    <tr>
      <th>820</th>
      <td>US-2014-166611</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>68.742</td>
      <td>68.7420</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>821</th>
      <td>US-2014-167920</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1827.51</td>
      <td>NaN</td>
      <td>1827.510</td>
      <td>NaN</td>
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
      <th>822</th>
      <td>US-2014-168116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8167.42</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8167.4200</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>823</th>
      <td>US-2014-168690</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.808</td>
      <td>2.808</td>
      <td>NaN</td>
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
      <th>824</th>
      <td>Grand Total</td>
      <td>20802.173</td>
      <td>8132.409</td>
      <td>49724.2545</td>
      <td>116922.1345</td>
      <td>195580.971</td>
      <td>20792.607</td>
      <td>9907.308</td>
      <td>35243.231</td>
      <td>55942.7865</td>
      <td>121885.9325</td>
      <td>7737.786</td>
      <td>2977.456</td>
      <td>8791.127</td>
      <td>54748.6325</td>
      <td>74255.0015</td>
    </tr>
  </tbody>
</table>
<p>825 rows × 16 columns</p>
</div>



![dirty_data_1](/images/data_cleaning/data_dirty_1.png)

Here we see the dirty data from Google Sheets, which corresponds to the data we have seened in the dataframe above

![clean_data_1](/images/data_cleaning/data_clean_1.PNG)

Here is our end goal, where the data has been rearranged and ready for further processing or visualization.


```python
df.columns
```




    Index(['Segment>>', 'Consumer', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4',
           'Consumer Total', 'Corporate', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
           'Corporate Total', 'Home Office', 'Unnamed: 12', 'Unnamed: 13',
           'Unnamed: 14', 'Home Office Total'],
          dtype='object')



We see that the original data contains 3 rows of headers, the topmost of which was read as the column names of our dataframe.


```python
df = df.drop(['Consumer Total','Corporate Total','Home Office Total'], axis=1)
df = df[:-1]
df
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
      <th>Segment&gt;&gt;</th>
      <th>Consumer</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Corporate</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Home Office</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
      <th>Unnamed: 14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ship Mode&gt;&gt;</td>
      <td>First Class</td>
      <td>Same Day</td>
      <td>Second Class</td>
      <td>Standard Class</td>
      <td>First Class</td>
      <td>Same Day</td>
      <td>Second Class</td>
      <td>Standard Class</td>
      <td>First Class</td>
      <td>Same Day</td>
      <td>Second Class</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order ID</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>CA-2011-100293</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-100706</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.44</td>
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
      <td>CA-2011-100895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>605.47</td>
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
    </tr>
    <tr>
      <th>819</th>
      <td>US-2014-166233</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
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
      <th>820</th>
      <td>US-2014-166611</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>68.742</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>821</th>
      <td>US-2014-167920</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1827.51</td>
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
      <th>822</th>
      <td>US-2014-168116</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8167.42</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>823</th>
      <td>US-2014-168690</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.808</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>824 rows × 13 columns</p>
</div>



We have removed the Subtotal columns and Grandtotal row from the dataset as we will not be including it in our end goal.


```python
col_names = ['Segment']
i = 1
for v in df.iloc[0,1:]:
    if i < 5:
        col_names.append('Consumer-' + v)
    elif i < 9:
        col_names.append('Corporate-' + v)
    else:
        col_names.append('Home Office-' + v)
    i += 1
col_names
```




    ['Segment',
     'Consumer-First Class',
     'Consumer-Same Day',
     'Consumer-Second Class',
     'Consumer-Standard Class',
     'Corporate-First Class',
     'Corporate-Same Day',
     'Corporate-Second Class',
     'Corporate-Standard Class',
     'Home Office-First Class',
     'Home Office-Same Day',
     'Home Office-Second Class',
     'Home Office-Standard Class']




```python
df = df.rename(columns = dict(zip(df.columns, col_names)))
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
      <th>Segment</th>
      <th>Consumer-First Class</th>
      <th>Consumer-Same Day</th>
      <th>Consumer-Second Class</th>
      <th>Consumer-Standard Class</th>
      <th>Corporate-First Class</th>
      <th>Corporate-Same Day</th>
      <th>Corporate-Second Class</th>
      <th>Corporate-Standard Class</th>
      <th>Home Office-First Class</th>
      <th>Home Office-Same Day</th>
      <th>Home Office-Second Class</th>
      <th>Home Office-Standard Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ship Mode&gt;&gt;</td>
      <td>First Class</td>
      <td>Same Day</td>
      <td>Second Class</td>
      <td>Standard Class</td>
      <td>First Class</td>
      <td>Same Day</td>
      <td>Second Class</td>
      <td>Standard Class</td>
      <td>First Class</td>
      <td>Same Day</td>
      <td>Second Class</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order ID</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>CA-2011-100293</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-100706</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.44</td>
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
      <td>CA-2011-100895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>605.47</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Here we have quickly put together new column names by concatenating the segment and shipmode of the order. As there are relatively few columns in this dataset, we have partially hardcoded the loop. Should there be significantly more columns, more optimization will be required to make use of the existing headers in the dataframe.

The column names are replaced with the newly created names, these names will be used when rearranging the data.


```python
# checking nulls in segment
df.Segment.isna().sum()
```




    0



We checked that there are no missing Order IDs.


```python
df = df[2:]
```

We removed the extra 2 rows of headers that we no longer require.


```python
# checking nulls for Order ID (every row except)
# expect 11 na per row (only 1 column will contain sales information)
df[2:].isna().sum(axis=1).value_counts()
```




    11    820
    Name: count, dtype: int64



Here we made sure that every Order ID only has 1 sale data within the 12 different Segment/ShippingMode combinations.


```python
sales_col = df.apply(pd.Series.last_valid_index, axis=1)
sales_col
```




    2      Home Office-Standard Class
    3           Consumer-Second Class
    4         Consumer-Standard Class
    5        Corporate-Standard Class
    6           Consumer-Second Class
                      ...            
    819       Consumer-Standard Class
    820      Corporate-Standard Class
    821         Consumer-Second Class
    822            Corporate-Same Day
    823       Consumer-Standard Class
    Length: 822, dtype: object



Then we extracted the Segment/ShippingMode combination for each Order ID by searching for the last valid index within each row (first valid index would return the Order ID, hence we look for last valid index here).


```python
cleaned_df = pd.DataFrame()
cleaned_df['Order ID'] = df['Segment']
cleaned_df['Segment_ShipMode'] = sales_col
cleaned_df.head()
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
      <th>Order ID</th>
      <th>Segment_ShipMode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>CA-2011-100293</td>
      <td>Home Office-Standard Class</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-100706</td>
      <td>Consumer-Second Class</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA-2011-100895</td>
      <td>Consumer-Standard Class</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CA-2011-100916</td>
      <td>Corporate-Standard Class</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CA-2011-101266</td>
      <td>Consumer-Second Class</td>
    </tr>
  </tbody>
</table>
</div>



Creating our final dataframe to store our "clean" data, we store the extracted Segment/ShippingMode for each Order ID within a new column.


```python
df['sales_col'] = sales_col
idx, cols = pd.factorize(df['sales_col'])
```

Over here we used pd.factorize() to encode our new column which we will use to extract the sales data to our new dataframe.


```python
cleaned_df['Sales'] = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
cleaned_df
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
      <th>Order ID</th>
      <th>Segment_ShipMode</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>CA-2011-100293</td>
      <td>Home Office-Standard Class</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-100706</td>
      <td>Consumer-Second Class</td>
      <td>129.44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA-2011-100895</td>
      <td>Consumer-Standard Class</td>
      <td>605.47</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CA-2011-100916</td>
      <td>Corporate-Standard Class</td>
      <td>788.86</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CA-2011-101266</td>
      <td>Consumer-Second Class</td>
      <td>13.36</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>819</th>
      <td>US-2014-166233</td>
      <td>Consumer-Standard Class</td>
      <td>24</td>
    </tr>
    <tr>
      <th>820</th>
      <td>US-2014-166611</td>
      <td>Corporate-Standard Class</td>
      <td>68.742</td>
    </tr>
    <tr>
      <th>821</th>
      <td>US-2014-167920</td>
      <td>Consumer-Second Class</td>
      <td>1827.51</td>
    </tr>
    <tr>
      <th>822</th>
      <td>US-2014-168116</td>
      <td>Corporate-Same Day</td>
      <td>8167.42</td>
    </tr>
    <tr>
      <th>823</th>
      <td>US-2014-168690</td>
      <td>Consumer-Standard Class</td>
      <td>2.808</td>
    </tr>
  </tbody>
</table>
<p>822 rows × 3 columns</p>
</div>



Using the encoded Segment_ShipMode column we could extract the sales data by indexing our original dataframe.


```python
cleaned_df[['Segment','Ship Mode']] = cleaned_df['Segment_ShipMode'].str.split('-', expand=True)
cleaned_df = cleaned_df.drop('Segment_ShipMode', axis=1)
```


```python
cleaned_df = cleaned_df[['Segment','Ship Mode','Order ID','Sales']]
cleaned_df = cleaned_df.sort_values(['Segment','Ship Mode','Order ID','Sales']).reset_index(drop=True)
cleaned_df.head()
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
      <th>Segment</th>
      <th>Ship Mode</th>
      <th>Order ID</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Consumer</td>
      <td>First Class</td>
      <td>CA-2011-103366</td>
      <td>149.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Consumer</td>
      <td>First Class</td>
      <td>CA-2011-109043</td>
      <td>243.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Consumer</td>
      <td>First Class</td>
      <td>CA-2011-113166</td>
      <td>9.568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Consumer</td>
      <td>First Class</td>
      <td>CA-2011-124023</td>
      <td>8.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Consumer</td>
      <td>First Class</td>
      <td>CA-2011-130155</td>
      <td>34.2</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we split the Segment_ShipMode columns into their respective columns and sort the values so that the final data is formatted similar to our end goal shown below in Google Sheets.

![cleaned_data_1](/images/data_cleaning/data_clean_1.PNG)

<a id='2'></a>
## 2. Badly Structured Sales Data 2
The second dataset is structured very similarly to our first, just with different orders for its headers and a date column instead of order id columns.


```python
data2 = pd.read_excel('2.-Badly-Structured-Sales-Data-2.xlsx')
df = data2.copy()
df
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
      <th>Ship Mode</th>
      <th>First Class</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Same Day</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Second Class</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Standard Class</th>
      <th>Unnamed: 11</th>
      <th>Unnamed: 12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Segment</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order Date</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>2013-03-14 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-16 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-06-02 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>605.47</td>
      <td>NaN</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>819</th>
      <td>2016-07-03 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>820</th>
      <td>2016-03-28 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>68.742</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>821</th>
      <td>2016-12-09 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1827.51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>822</th>
      <td>2016-11-04 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8167.42</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>823</th>
      <td>2016-01-08 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.808</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>824 rows × 13 columns</p>
</div>



![dirty_data_2](/images/data_cleaning/data_dirty_2.PNG)

Above we see the "dirty" data in Google Sheets. Below we see our end goal, our Order Date will not be in exactly the same format, opting to leave it in datetime format for flexibility when exporting the cleaned data through Pandas.

![clean_data_2](/images/data_cleaning/data_clean_2.PNG)


```python
col_names = ['Order Date']
suff = ['First Class','Same Day','Second Class','Standard Class']
i = 0
for v in df.iloc[0,1:]:
    col_names.append(suff[i//3] + '-' + v)
    i += 1
col_names
```




    ['Order Date',
     'First Class-Consumer',
     'First Class-Corporate',
     'First Class-Home Office',
     'Same Day-Consumer',
     'Same Day-Corporate',
     'Same Day-Home Office',
     'Second Class-Consumer',
     'Second Class-Corporate',
     'Second Class-Home Office',
     'Standard Class-Consumer',
     'Standard Class-Corporate',
     'Standard Class-Home Office']



Next we create our list of new column names, this time opting for a slightly different approach to our code that is more reusable than the one we used for the dataset 1.


```python
df = df.rename(columns = dict(zip(df.columns, col_names)))
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
      <th>Order Date</th>
      <th>First Class-Consumer</th>
      <th>First Class-Corporate</th>
      <th>First Class-Home Office</th>
      <th>Same Day-Consumer</th>
      <th>Same Day-Corporate</th>
      <th>Same Day-Home Office</th>
      <th>Second Class-Consumer</th>
      <th>Second Class-Corporate</th>
      <th>Second Class-Home Office</th>
      <th>Standard Class-Consumer</th>
      <th>Standard Class-Corporate</th>
      <th>Standard Class-Home Office</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Segment</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order Date</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>2013-03-14 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-16 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2013-06-02 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>605.47</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Order Date'].isna().sum()
```




    0




```python
df[2:].isna().sum(axis=1).value_counts()
```




    11    822
    Name: count, dtype: int64



Next we rename our columns with our newly created column names and conduct our sanity checks on our data integrity, confirming that the sales data corresponds to order date properly.


```python
def extract_sales(df):
    sales_col = df.apply(pd.Series.last_valid_index, axis=1)
    df['sales_col'] = sales_col
    idx, cols = pd.factorize(df['sales_col'])
    extracted_sales = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
    return extracted_sales, sales_col
extracted_sales, sales_col = extract_sales(df)
```


```python
cleaned_df = pd.DataFrame()
cleaned_df['sales_col'] = sales_col
cleaned_df[['Ship Mode', 'Segment']] = cleaned_df['sales_col'].str.split('-',expand=True)
cleaned_df["Order Date"] = df['Order Date']
cleaned_df['Sales'] = extracted_sales
cleaned_df
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
      <th>sales_col</th>
      <th>Ship Mode</th>
      <th>Segment</th>
      <th>Order Date</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Standard Class-Home Office</td>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>Segment</td>
      <td>Home Office</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order Date</td>
      <td>Order Date</td>
      <td>None</td>
      <td>Order Date</td>
      <td>Order Date</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Standard Class-Home Office</td>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>2013-03-14 00:00:00</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Second Class-Consumer</td>
      <td>Second Class</td>
      <td>Consumer</td>
      <td>2013-12-16 00:00:00</td>
      <td>129.44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Standard Class-Consumer</td>
      <td>Standard Class</td>
      <td>Consumer</td>
      <td>2013-06-02 00:00:00</td>
      <td>605.47</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Standard Class-Consumer</td>
      <td>Standard Class</td>
      <td>Consumer</td>
      <td>2016-07-03 00:00:00</td>
      <td>24</td>
    </tr>
    <tr>
      <th>820</th>
      <td>Standard Class-Corporate</td>
      <td>Standard Class</td>
      <td>Corporate</td>
      <td>2016-03-28 00:00:00</td>
      <td>68.742</td>
    </tr>
    <tr>
      <th>821</th>
      <td>Second Class-Consumer</td>
      <td>Second Class</td>
      <td>Consumer</td>
      <td>2016-12-09 00:00:00</td>
      <td>1827.51</td>
    </tr>
    <tr>
      <th>822</th>
      <td>Same Day-Corporate</td>
      <td>Same Day</td>
      <td>Corporate</td>
      <td>2016-11-04 00:00:00</td>
      <td>8167.42</td>
    </tr>
    <tr>
      <th>823</th>
      <td>Standard Class-Consumer</td>
      <td>Standard Class</td>
      <td>Consumer</td>
      <td>2016-01-08 00:00:00</td>
      <td>2.808</td>
    </tr>
  </tbody>
</table>
<p>824 rows × 5 columns</p>
</div>




```python
cleaned_df = cleaned_df[2:][cleaned_df.columns[1:]].sort_values(['Ship Mode','Segment']).reset_index(drop=True)
cleaned_df
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
      <th>Ship Mode</th>
      <th>Segment</th>
      <th>Order Date</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>2013-01-15 00:00:00</td>
      <td>149.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>2013-08-15 00:00:00</td>
      <td>243.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>2013-12-24 00:00:00</td>
      <td>9.568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>2013-04-07 00:00:00</td>
      <td>8.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>2013-05-19 00:00:00</td>
      <td>34.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>817</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>2016-03-17 00:00:00</td>
      <td>4.608</td>
    </tr>
    <tr>
      <th>818</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>2016-04-23 00:00:00</td>
      <td>513.496</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>2016-05-27 00:00:00</td>
      <td>598.31</td>
    </tr>
    <tr>
      <th>820</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>2016-09-24 00:00:00</td>
      <td>148.16</td>
    </tr>
    <tr>
      <th>821</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>2016-11-04 00:00:00</td>
      <td>9.664</td>
    </tr>
  </tbody>
</table>
<p>822 rows × 4 columns</p>
</div>



Utilizing the same workflow as dataset 1, we achieve the final "clean" dataset as shown above, which corresponds to our end goal as seen below in Google Sheets.

![cleaned_data_2](/images/data_cleaning/data_clean_2.PNG)

<a id='3'></a>
## 3. Badly Structured Sales Data 3
This third dataset seems like a combination of our first two, and we will working towards an end goal of 5 columns in a long format. Similarly, we will be leaving our date in datetime format for flexibility when exporting the data.


```python
data3 = pd.read_excel('3.-Badly-Structured-Sales-Data-3.xlsx')
df = data3.copy()
df
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
      <th>Unnamed: 0</th>
      <th>Ship Mode</th>
      <th>First Class</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Same Day</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Second Class</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Standard Class</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Segment</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order ID</td>
      <td>Order Date</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>CA-2011-100293</td>
      <td>2013-03-14 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-100706</td>
      <td>2013-12-16 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA-2011-100895</td>
      <td>2013-06-02 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>605.47</td>
      <td>NaN</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>820</th>
      <td>US-2014-166611</td>
      <td>2016-03-28 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>68.742</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>821</th>
      <td>US-2014-167920</td>
      <td>2016-12-09 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1827.51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>822</th>
      <td>US-2014-168116</td>
      <td>2016-11-04 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8167.42</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>823</th>
      <td>US-2014-168690</td>
      <td>2016-01-08 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.808</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>824</th>
      <td>Grand Total</td>
      <td>NaN</td>
      <td>20802.173</td>
      <td>20792.607</td>
      <td>7737.786</td>
      <td>8132.409</td>
      <td>9907.308</td>
      <td>2977.456</td>
      <td>49724.2545</td>
      <td>35243.231</td>
      <td>8791.127</td>
      <td>116922.1345</td>
      <td>55942.7865</td>
      <td>54748.6325</td>
    </tr>
  </tbody>
</table>
<p>825 rows × 14 columns</p>
</div>



![dirty_data_3](/images/data_cleaning/data_dirty_3.png)

![cleaned_data_3](/images/data_cleaning/data_clean_3.PNG)


```python
df[df['Ship Mode'].isna()]
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
      <th>Unnamed: 0</th>
      <th>Ship Mode</th>
      <th>First Class</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Same Day</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Second Class</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Standard Class</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>824</th>
      <td>Grand Total</td>
      <td>NaN</td>
      <td>20802.173</td>
      <td>20792.607</td>
      <td>7737.786</td>
      <td>8132.409</td>
      <td>9907.308</td>
      <td>2977.456</td>
      <td>49724.2545</td>
      <td>35243.231</td>
      <td>8791.127</td>
      <td>116922.1345</td>
      <td>55942.7865</td>
      <td>54748.6325</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df[:-1]
df.tail(3)
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
      <th>Unnamed: 0</th>
      <th>Ship Mode</th>
      <th>First Class</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Same Day</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Second Class</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Standard Class</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>821</th>
      <td>US-2014-167920</td>
      <td>2016-12-09 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1827.51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>822</th>
      <td>US-2014-168116</td>
      <td>2016-11-04 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8167.42</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>823</th>
      <td>US-2014-168690</td>
      <td>2016-01-08 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.808</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



First we checked for the grand total rows and removed them from the dataframe.


```python
col_names = ['Order ID','Order Date']
suff = ['First Class','Same Day','Second Class','Standard Class']
i = 0
for v in df.iloc[0,2:]:
    col_names.append(suff[i//3] + '-' + v)
    i += 1
col_names
```




    ['Order ID',
     'Order Date',
     'First Class-Consumer',
     'First Class-Corporate',
     'First Class-Home Office',
     'Same Day-Consumer',
     'Same Day-Corporate',
     'Same Day-Home Office',
     'Second Class-Consumer',
     'Second Class-Corporate',
     'Second Class-Home Office',
     'Standard Class-Consumer',
     'Standard Class-Corporate',
     'Standard Class-Home Office']




```python
df = df.rename(columns = dict(zip(df.columns, col_names)))
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
      <th>Order ID</th>
      <th>Order Date</th>
      <th>First Class-Consumer</th>
      <th>First Class-Corporate</th>
      <th>First Class-Home Office</th>
      <th>Same Day-Consumer</th>
      <th>Same Day-Corporate</th>
      <th>Same Day-Home Office</th>
      <th>Second Class-Consumer</th>
      <th>Second Class-Corporate</th>
      <th>Second Class-Home Office</th>
      <th>Standard Class-Consumer</th>
      <th>Standard Class-Corporate</th>
      <th>Standard Class-Home Office</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Segment</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order ID</td>
      <td>Order Date</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>CA-2011-100293</td>
      <td>2013-03-14 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-100706</td>
      <td>2013-12-16 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA-2011-100895</td>
      <td>2013-06-02 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>605.47</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Order ID','Order Date']].isna().sum()
```




    Order ID      1
    Order Date    0
    dtype: int64




```python
df[2:].isna().sum(axis=1).value_counts()
```




    11    822
    Name: count, dtype: int64



Then we renamed the dataframe header with our newly created column names and completed our sanity checks on the data integrity. The single missing value from "Order ID" column stems from one of the extra headers columns containing a NaN value and can be safely ignored.


```python
def extract_sales(df):
    sales_col = df.apply(pd.Series.last_valid_index, axis=1)
    df['sales_col'] = sales_col
    idx, cols = pd.factorize(df['sales_col'])
    extracted_sales = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
    return extracted_sales, sales_col
extracted_sales, sales_col = extract_sales(df)
```


```python
cleaned_df = pd.DataFrame()
cleaned_df['sales_col'] = sales_col
cleaned_df[['Ship Mode', 'Segment']] = cleaned_df['sales_col'].str.split('-',expand=True)
cleaned_df["Order ID"] = df['Order ID']
cleaned_df["Order Date"] = df['Order Date']
cleaned_df['Sales'] = extracted_sales
cleaned_df
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
      <th>sales_col</th>
      <th>Ship Mode</th>
      <th>Segment</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Standard Class-Home Office</td>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>NaN</td>
      <td>Segment</td>
      <td>Home Office</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order Date</td>
      <td>Order Date</td>
      <td>None</td>
      <td>Order ID</td>
      <td>Order Date</td>
      <td>Order Date</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Standard Class-Home Office</td>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>CA-2011-100293</td>
      <td>2013-03-14 00:00:00</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Second Class-Consumer</td>
      <td>Second Class</td>
      <td>Consumer</td>
      <td>CA-2011-100706</td>
      <td>2013-12-16 00:00:00</td>
      <td>129.44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Standard Class-Consumer</td>
      <td>Standard Class</td>
      <td>Consumer</td>
      <td>CA-2011-100895</td>
      <td>2013-06-02 00:00:00</td>
      <td>605.47</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Standard Class-Consumer</td>
      <td>Standard Class</td>
      <td>Consumer</td>
      <td>US-2014-166233</td>
      <td>2016-07-03 00:00:00</td>
      <td>24</td>
    </tr>
    <tr>
      <th>820</th>
      <td>Standard Class-Corporate</td>
      <td>Standard Class</td>
      <td>Corporate</td>
      <td>US-2014-166611</td>
      <td>2016-03-28 00:00:00</td>
      <td>68.742</td>
    </tr>
    <tr>
      <th>821</th>
      <td>Second Class-Consumer</td>
      <td>Second Class</td>
      <td>Consumer</td>
      <td>US-2014-167920</td>
      <td>2016-12-09 00:00:00</td>
      <td>1827.51</td>
    </tr>
    <tr>
      <th>822</th>
      <td>Same Day-Corporate</td>
      <td>Same Day</td>
      <td>Corporate</td>
      <td>US-2014-168116</td>
      <td>2016-11-04 00:00:00</td>
      <td>8167.42</td>
    </tr>
    <tr>
      <th>823</th>
      <td>Standard Class-Consumer</td>
      <td>Standard Class</td>
      <td>Consumer</td>
      <td>US-2014-168690</td>
      <td>2016-01-08 00:00:00</td>
      <td>2.808</td>
    </tr>
  </tbody>
</table>
<p>824 rows × 6 columns</p>
</div>




```python
cleaned_df = cleaned_df[2:][cleaned_df.columns[1:]].sort_values(['Ship Mode', 'Segment', 'Order ID', 'Order Date']).reset_index(drop=True)
cleaned_df
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
      <th>Ship Mode</th>
      <th>Segment</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-103366</td>
      <td>2013-01-15 00:00:00</td>
      <td>149.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-109043</td>
      <td>2013-08-15 00:00:00</td>
      <td>243.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-113166</td>
      <td>2013-12-24 00:00:00</td>
      <td>9.568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-124023</td>
      <td>2013-04-07 00:00:00</td>
      <td>8.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-130155</td>
      <td>2013-05-19 00:00:00</td>
      <td>34.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>817</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-129224</td>
      <td>2016-03-17 00:00:00</td>
      <td>4.608</td>
    </tr>
    <tr>
      <th>818</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-132031</td>
      <td>2016-04-23 00:00:00</td>
      <td>513.496</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-132297</td>
      <td>2016-05-27 00:00:00</td>
      <td>598.31</td>
    </tr>
    <tr>
      <th>820</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-132675</td>
      <td>2016-09-24 00:00:00</td>
      <td>148.16</td>
    </tr>
    <tr>
      <th>821</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-156083</td>
      <td>2016-11-04 00:00:00</td>
      <td>9.664</td>
    </tr>
  </tbody>
</table>
<p>822 rows × 5 columns</p>
</div>



Again, using the same workflow as datasets 1 and 2, we rearranged the data into our "clean" dataset shown above. This corresponds nicely to our target final state as shown below in Google Sheets.

![cleaned_data_3](/images/data_cleaning/data_clean_3.PNG)

<a id='4'></a>
## 4. Badly Structured Sales Data 4
According to the source this dataset is very similar to data 3 "with a little different flavor".


```python
data4 = pd.read_excel('4.-Badly-Structured-Sales-Data-4.xlsx')
df = data4.copy()
df
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
      <th>Unnamed: 0</th>
      <th>Ship Mode</th>
      <th>First Class</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Same Day</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Second Class</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Standard Class</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Segment</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order ID</td>
      <td>Order Date</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>CA-2011-100293</td>
      <td>2013-03-14 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-100706</td>
      <td>2013-12-16 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA-2011-100895</td>
      <td>2013-06-02 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>605.47</td>
      <td>NaN</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>820</th>
      <td>US-2014-166611</td>
      <td>2016-03-28 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>68.742</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>821</th>
      <td>US-2014-167920</td>
      <td>2016-12-09 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1827.51</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>822</th>
      <td>US-2014-168116</td>
      <td>2016-11-04 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8167.42</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>823</th>
      <td>US-2014-168690</td>
      <td>2016-01-08 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.808</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>824</th>
      <td>NaN</td>
      <td>Grand Total</td>
      <td>20802.173</td>
      <td>20792.607</td>
      <td>7737.786</td>
      <td>8132.409</td>
      <td>9907.308</td>
      <td>2977.456</td>
      <td>49724.2545</td>
      <td>35243.231</td>
      <td>8791.127</td>
      <td>116922.1345</td>
      <td>55942.7865</td>
      <td>54748.6325</td>
    </tr>
  </tbody>
</table>
<p>825 rows × 14 columns</p>
</div>



![dirty_data_4](/images/data_cleaning/data_dirty_4.PNG)
![cleaned_data_4](/images/data_cleaning/data_clean_4.PNG)

Looking at the "dirty" and "clean" forms of dataset 4 I realised that visually I could not discern any difference from dataset 3. Hence, we shall compare the data using pandas to detect the variations in both dataframes.


```python
merged_df = data3.merge(data4, indicator=True, how='outer')
changed_rows = merged_df[merged_df['_merge'] != 'both']
changed_rows.drop('_merge',axis=1)
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
      <th>Unnamed: 0</th>
      <th>Ship Mode</th>
      <th>First Class</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Same Day</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Second Class</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Standard Class</th>
      <th>Unnamed: 12</th>
      <th>Unnamed: 13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>688</th>
      <td>Grand Total</td>
      <td>NaN</td>
      <td>20802.173</td>
      <td>20792.607</td>
      <td>7737.786</td>
      <td>8132.409</td>
      <td>9907.308</td>
      <td>2977.456</td>
      <td>49724.2545</td>
      <td>35243.231</td>
      <td>8791.127</td>
      <td>116922.1345</td>
      <td>55942.7865</td>
      <td>54748.6325</td>
    </tr>
    <tr>
      <th>824</th>
      <td>NaN</td>
      <td>Grand Total</td>
      <td>20802.173</td>
      <td>20792.607</td>
      <td>7737.786</td>
      <td>8132.409</td>
      <td>9907.308</td>
      <td>2977.456</td>
      <td>49724.2545</td>
      <td>35243.231</td>
      <td>8791.127</td>
      <td>116922.1345</td>
      <td>55942.7865</td>
      <td>54748.6325</td>
    </tr>
  </tbody>
</table>
</div>



From the above we found that the only difference between dataset 3 and 4 is the final row of Grand Total values. Functionally this will not result in any difference to how we will treat the dataset in Pandas as we have been removing this row in all of the previous sections, so feel free to skip this section as I will be using the same procedure as the previous section to clean dataset 4.


```python
col_names = ['Order ID','Order Date']
suff = ['First Class','Same Day','Second Class','Standard Class']
i = 0
for v in df.iloc[0,2:]:
    col_names.append(suff[i//3] + '-' + v)
    i += 1
col_names
```




    ['Order ID',
     'Order Date',
     'First Class-Consumer',
     'First Class-Corporate',
     'First Class-Home Office',
     'Same Day-Consumer',
     'Same Day-Corporate',
     'Same Day-Home Office',
     'Second Class-Consumer',
     'Second Class-Corporate',
     'Second Class-Home Office',
     'Standard Class-Consumer',
     'Standard Class-Corporate',
     'Standard Class-Home Office']




```python
df = df.rename(columns = dict(zip(df.columns, col_names)))
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
      <th>Order ID</th>
      <th>Order Date</th>
      <th>First Class-Consumer</th>
      <th>First Class-Corporate</th>
      <th>First Class-Home Office</th>
      <th>Same Day-Consumer</th>
      <th>Same Day-Corporate</th>
      <th>Same Day-Home Office</th>
      <th>Second Class-Consumer</th>
      <th>Second Class-Corporate</th>
      <th>Second Class-Home Office</th>
      <th>Standard Class-Consumer</th>
      <th>Standard Class-Corporate</th>
      <th>Standard Class-Home Office</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Segment</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
      <td>Consumer</td>
      <td>Corporate</td>
      <td>Home Office</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order ID</td>
      <td>Order Date</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <td>CA-2011-100293</td>
      <td>2013-03-14 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-100706</td>
      <td>2013-12-16 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>129.44</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA-2011-100895</td>
      <td>2013-06-02 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>605.47</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['Order ID','Order Date']].isna().sum()
```




    Order ID      2
    Order Date    0
    dtype: int64




```python
df[2:].isna().sum(axis=1).value_counts()
```




    11    822
    1       1
    Name: count, dtype: int64




```python
df = df[:-1]
```


```python
def extract_sales(df):
    sales_col = df.apply(pd.Series.last_valid_index, axis=1)
    df['sales_col'] = sales_col
    idx, cols = pd.factorize(df['sales_col'])
    extracted_sales = df.reindex(cols, axis=1).to_numpy()[np.arange(len(df)), idx]
    return extracted_sales, sales_col
extracted_sales, sales_col = extract_sales(df)
```


```python
cleaned_df = pd.DataFrame()
cleaned_df['sales_col'] = sales_col
cleaned_df[['Ship Mode', 'Segment']] = cleaned_df['sales_col'].str.split('-',expand=True)
cleaned_df["Order ID"] = df['Order ID']
cleaned_df["Order Date"] = df['Order Date']
cleaned_df['Sales'] = extracted_sales
cleaned_df
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
      <th>sales_col</th>
      <th>Ship Mode</th>
      <th>Segment</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Standard Class-Home Office</td>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>NaN</td>
      <td>Segment</td>
      <td>Home Office</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Order Date</td>
      <td>Order Date</td>
      <td>None</td>
      <td>Order ID</td>
      <td>Order Date</td>
      <td>Order Date</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Standard Class-Home Office</td>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>CA-2011-100293</td>
      <td>2013-03-14 00:00:00</td>
      <td>91.056</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Second Class-Consumer</td>
      <td>Second Class</td>
      <td>Consumer</td>
      <td>CA-2011-100706</td>
      <td>2013-12-16 00:00:00</td>
      <td>129.44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Standard Class-Consumer</td>
      <td>Standard Class</td>
      <td>Consumer</td>
      <td>CA-2011-100895</td>
      <td>2013-06-02 00:00:00</td>
      <td>605.47</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Standard Class-Consumer</td>
      <td>Standard Class</td>
      <td>Consumer</td>
      <td>US-2014-166233</td>
      <td>2016-07-03 00:00:00</td>
      <td>24</td>
    </tr>
    <tr>
      <th>820</th>
      <td>Standard Class-Corporate</td>
      <td>Standard Class</td>
      <td>Corporate</td>
      <td>US-2014-166611</td>
      <td>2016-03-28 00:00:00</td>
      <td>68.742</td>
    </tr>
    <tr>
      <th>821</th>
      <td>Second Class-Consumer</td>
      <td>Second Class</td>
      <td>Consumer</td>
      <td>US-2014-167920</td>
      <td>2016-12-09 00:00:00</td>
      <td>1827.51</td>
    </tr>
    <tr>
      <th>822</th>
      <td>Same Day-Corporate</td>
      <td>Same Day</td>
      <td>Corporate</td>
      <td>US-2014-168116</td>
      <td>2016-11-04 00:00:00</td>
      <td>8167.42</td>
    </tr>
    <tr>
      <th>823</th>
      <td>Standard Class-Consumer</td>
      <td>Standard Class</td>
      <td>Consumer</td>
      <td>US-2014-168690</td>
      <td>2016-01-08 00:00:00</td>
      <td>2.808</td>
    </tr>
  </tbody>
</table>
<p>824 rows × 6 columns</p>
</div>




```python
cleaned_df = cleaned_df[2:][cleaned_df.columns[1:]].sort_values(['Ship Mode', 'Segment', 'Order ID', 'Order Date']).reset_index(drop=True)
cleaned_df
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
      <th>Ship Mode</th>
      <th>Segment</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-103366</td>
      <td>2013-01-15 00:00:00</td>
      <td>149.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-109043</td>
      <td>2013-08-15 00:00:00</td>
      <td>243.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-113166</td>
      <td>2013-12-24 00:00:00</td>
      <td>9.568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-124023</td>
      <td>2013-04-07 00:00:00</td>
      <td>8.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>First Class</td>
      <td>Consumer</td>
      <td>CA-2011-130155</td>
      <td>2013-05-19 00:00:00</td>
      <td>34.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>817</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-129224</td>
      <td>2016-03-17 00:00:00</td>
      <td>4.608</td>
    </tr>
    <tr>
      <th>818</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-132031</td>
      <td>2016-04-23 00:00:00</td>
      <td>513.496</td>
    </tr>
    <tr>
      <th>819</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-132297</td>
      <td>2016-05-27 00:00:00</td>
      <td>598.31</td>
    </tr>
    <tr>
      <th>820</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-132675</td>
      <td>2016-09-24 00:00:00</td>
      <td>148.16</td>
    </tr>
    <tr>
      <th>821</th>
      <td>Standard Class</td>
      <td>Home Office</td>
      <td>US-2014-156083</td>
      <td>2016-11-04 00:00:00</td>
      <td>9.664</td>
    </tr>
  </tbody>
</table>
<p>822 rows × 5 columns</p>
</div>



![cleaned_data_4](/images/data_cleaning/data_clean_4.PNG)

As seen above, the "clean" dataset and the target final state corresponds to one another. We have cleaned the first 4 of 8 datasets, exploring ways to rearrange badly structured data through Pandas. In this notebook some repetitive steps could be observed due to the similar nature some of these datasets were structured, but in Part 2 there will be more variation to our data cleaning steps as we explore datasets that have their values poorly entered, requiring us to devise ways to identify and rectify the dirty data.
