---
layout: post
title:  "Dirty Data Samples - Cleaning Data With Pandas - Part 2"
date:   2024-04-05 17:57:09 +0800
category: [data_wrangling, misc]
tag: [numpy, pandas, fun]
summary: "In this notebook we will explore 4 dirty datasets sourced from the internet that has data incorrectly recorded and clean them using Pandas package in Python."
image: /images/banners/data_cleaning.png
---

## Contents
0. [Introduction](#0)
1. [Jumbled Customer Details](#1)
2. [Medicine Data with Combined Quantity and Measure](#2)
3. [Hospital Data with Mixed Numbers and Characters](#3)
4. [Invoices with Merged Categories and Merged Amounts](#4)

<a id='0'></a>
## 0. Introduction
In this notebook we will explore 4 "dirty" datasets from a [power bi resources website](https://foresightbi.com.ng/microsoft-power-bi/dirty-data-samples-to-practice-on/) containing poorly recorded data that needs to be rectified. Utilizing pandas we will fix the errors within the dataset so that they are ready for further processing/exploration. The website also shows the final requirements that they want the "clean" dataset to look like, and we will be working with that in mind.

This is part two of two notebooks working with these "dirty" datasets. In [Part 1](/data_wrangling/misc/2024/04/04/data_cleaning_part1.html) we have looked at another 4 datasets that contains data that have poorly structured.


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




```python
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

<a id='1'></a>
## 1. Jumbled Customer Details
In dataset 5 we will explore a dataset that has all of their values stored in one column. This is encountered often when downloading or copying something from the internet, or when webscraping.


```python
data5 = pd.read_excel('5.-Jumbled-up-Customers-Details.xlsx', header=None)
df = data5.copy()
df = df.rename(columns={0:"Strings"})
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
      <th>Strings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Name Hussein Hakeem Address Number 22 Fioye Cr...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Name Arojoye Samuel Address 11 Omolade Close O...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Name Alex Ezurum Address 1 Adamu Lane, Abuja A...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Name Susan Nwaimo Address Number 58 Yaba Stree...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Name Ajao Opeyemi Address No12 Olubunmi Street...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Name Banjoko Adebusola Address 34 Ngige Street...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Name Muhammed Olabisi Address 13, ICAN road, E...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Name Oluwagbemi Mojisola Address ACCA Lane, On...</td>
    </tr>
  </tbody>
</table>
</div>



![dirty_data_5](/images/data_cleaning/data_dirty_5.PNG)
![cleaned_data_5](/images/data_cleaning/data_clean_5.PNG)

From the above we see that our goal for this dataset is to recognise and separate the original data into their respective columns, to do that we will be making use of regular expression and some pandas functions to transform our data.


```python
pattern = '|'.join(['Name', 'Address','Age','Gender'])
pattern
```




    'Name|Address|Age|Gender'




```python
df['Strings'] = df['Strings'].str.replace(pattern, '|', regex=True)
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
      <th>Strings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>| Hussein Hakeem | Number 22 Fioye Crescent Su...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>| Arojoye Samuel | 11 Omolade Close Omole Esta...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>| Alex Ezurum | 1 Adamu Lane, Abuja | 14 | Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>| Susan Nwaimo | Number 58 Yaba Street, Kaduna...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>| Ajao Opeyemi | No12 Olubunmi Street, Abeokut...</td>
    </tr>
  </tbody>
</table>
</div>



Here we have replaced the categories/columns with a | symbol to properly distinguish the different categories in each row of the dataset.


```python
cleaned_df = pd.DataFrame()
cleaned_df[['na','Name','Address','Age','Gender']] = df['Strings'].str.split('|', expand=True)
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
      <th>na</th>
      <th>Name</th>
      <th>Address</th>
      <th>Age</th>
      <th>Gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>Hussein Hakeem</td>
      <td>Number 22 Fioye Crescent Surulere Lagos</td>
      <td>17</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>Arojoye Samuel</td>
      <td>11 Omolade Close Omole Estate Lagos</td>
      <td>16</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>Alex Ezurum</td>
      <td>1 Adamu Lane, Abuja</td>
      <td>14</td>
      <td>Male</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>Susan Nwaimo</td>
      <td>Number 58 Yaba Street, Kaduna State</td>
      <td>16</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>Ajao Opeyemi</td>
      <td>No12 Olubunmi Street, Abeokuta</td>
      <td>18</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>Banjoko Adebusola</td>
      <td>34 Ngige Street, Ugheli, Delta</td>
      <td>14</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>Muhammed Olabisi</td>
      <td>13, ICAN road, Enugu</td>
      <td>12</td>
      <td>Female</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>Oluwagbemi Mojisola</td>
      <td>ACCA Lane, Onitsha</td>
      <td>13</td>
      <td>Female</td>
    </tr>
  </tbody>
</table>
</div>



Then we can simply split them into their respective columns achieving our target final state!

![cleaned_data_5](/images/data_cleaning/data_clean_5.PNG)

<a id='2'></a>
## 2. Medicine Data with Combined Quantity and Measure
In this dataset we see the "Quantity" column records a mix of quantity and unit measure. Our target final state will be to split this into two columns, to verify that the data cleaning was completed correctly the sum of our final "Quantity" column should be 17600.0.


```python
data6 = pd.read_excel('6.-Hospital-Data-with-Mixed-Numbers-and-Characters.xlsx')
df = data6.copy()
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
      <th>Description</th>
      <th>Quantity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lotion Benzylbenzoate lotion</td>
      <td>0Bottle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Methylated spirit 100ml</td>
      <td>0Bottle</td>
    </tr>
    <tr>
      <th>2</th>
      <td>susp Magnessium Trisilicate 200ml</td>
      <td>0Bottle</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Susp. Amoxicillin 125mg/5ml</td>
      <td>0Bottle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Susp. Erythromycin 125mg/5ml</td>
      <td>0Bottle</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2274</th>
      <td>Syp Ascorbic acid</td>
      <td>20Bottle</td>
    </tr>
    <tr>
      <th>2275</th>
      <td>syr Cough Syrup (P) 100ml</td>
      <td>20Bottle</td>
    </tr>
    <tr>
      <th>2276</th>
      <td>syr Cough Syrup (A) 100ml</td>
      <td>10Bottle</td>
    </tr>
    <tr>
      <th>2277</th>
      <td>Cotton Wool 100g</td>
      <td>1Pcs</td>
    </tr>
    <tr>
      <th>2278</th>
      <td>Disposable gloves</td>
      <td>1Pairs</td>
    </tr>
  </tbody>
</table>
<p>2279 rows × 2 columns</p>
</div>



![dirty_data_6](/images/data_cleaning/data_dirty_6.PNG)
![cleaned_data_6](/images/data_cleaning/data_clean_6.PNG)


```python
cleaned_df = pd.DataFrame()
cleaned_df['Description'] = df['Description']
cleaned_df[['na1','Quantity','na2','Measure','na3']] = df['Quantity'].str.split('(\d+(\.\d*)?)([A-Za-z]+)', expand=True)
cleaned_df = cleaned_df.drop(['na1','na2','na3'], axis=1)
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
      <th>Description</th>
      <th>Quantity</th>
      <th>Measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lotion Benzylbenzoate lotion</td>
      <td>0</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Methylated spirit 100ml</td>
      <td>0</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>2</th>
      <td>susp Magnessium Trisilicate 200ml</td>
      <td>0</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Susp. Amoxicillin 125mg/5ml</td>
      <td>0</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Susp. Erythromycin 125mg/5ml</td>
      <td>0</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2274</th>
      <td>Syp Ascorbic acid</td>
      <td>20</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>2275</th>
      <td>syr Cough Syrup (P) 100ml</td>
      <td>20</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>2276</th>
      <td>syr Cough Syrup (A) 100ml</td>
      <td>10</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>2277</th>
      <td>Cotton Wool 100g</td>
      <td>1</td>
      <td>Pcs</td>
    </tr>
    <tr>
      <th>2278</th>
      <td>Disposable gloves</td>
      <td>1</td>
      <td>Pairs</td>
    </tr>
  </tbody>
</table>
<p>2279 rows × 3 columns</p>
</div>



Once again, with the help of regex we managed to split our original "Quantity" column into a column of float quantities "(\d+(\.\d*)?)" and a column of alphabetic unit measures "([A-Za-z]+)".


```python
cleaned_df['Quantity'] = cleaned_df['Quantity'].astype(float).round(2)
cleaned_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2279 entries, 0 to 2278
    Data columns (total 3 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Description  2279 non-null   object 
     1   Quantity     2279 non-null   float64
     2   Measure      2279 non-null   object 
    dtypes: float64(1), object(2)
    memory usage: 53.5+ KB
    


```python
cleaned_df.Quantity.sum()
```




    17600.0



Sanity check after splitting the columns show that non of the rows contain null value, and as expected the sum of the Quantity column is 17600.0.

Below we have a final look at the "cleaned" dataframe and the final state as seen in Google Sheets.


```python
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
      <th>Description</th>
      <th>Quantity</th>
      <th>Measure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lotion Benzylbenzoate lotion</td>
      <td>0.0</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Methylated spirit 100ml</td>
      <td>0.0</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>2</th>
      <td>susp Magnessium Trisilicate 200ml</td>
      <td>0.0</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Susp. Amoxicillin 125mg/5ml</td>
      <td>0.0</td>
      <td>Bottle</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Susp. Erythromycin 125mg/5ml</td>
      <td>0.0</td>
      <td>Bottle</td>
    </tr>
  </tbody>
</table>
</div>



![cleaned_data_6](/images/data_cleaning/data_clean_6.PNG)

<a id='3'></a>
## 3. Hospital Data with Mixed Numbers and Characters
In this dataset some of the data has letters used in place of some numbers, our goal would be to identify and convert these letters into their corresponding numbers.


```python
data7 = pd.read_excel('7.-Medicine-Data-with-lumped-Quantity-and-Measure.xlsx')
df = data7.copy()
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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tab. Cefuroxime 250mg</td>
      <td>10</td>
      <td>1000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tab. Cefuroxime 250mg</td>
      <td>10</td>
      <td>1000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tab. Cefuroxime 250mg</td>
      <td>10</td>
      <td>1000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cap Ampicillin</td>
      <td>100</td>
      <td>350</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cap Ampicillin</td>
      <td>100</td>
      <td>350</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
      <th>5841</th>
      <td>Inj.Vitamin B Complex</td>
      <td>Vial</td>
      <td>145</td>
      <td>30</td>
      <td>4350</td>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>5842</th>
      <td>Insulin (Actrapid)</td>
      <td>Vial</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5843</th>
      <td>Insulin (Insulutard)</td>
      <td>Vial</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5844</th>
      <td>Inj Amoxicillin 500mg</td>
      <td>vials</td>
      <td>54</td>
      <td>100</td>
      <td>79</td>
      <td>0</td>
      <td>75</td>
    </tr>
    <tr>
      <th>5845</th>
      <td>Inj Amoxy/clav 1.2</td>
      <td>vials</td>
      <td>7</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5846 rows × 7 columns</p>
</div>



![dirty_data_7](/images/data_cleaning/data_dirty_7.PNG)
![cleaned_data_7](/images/data_cleaning/data_clean_7.PNG)

From the above we see certain errors like using o in place of 0 and i in place of 1. Using python we will look through the entire dataset to identify rows where such an error is present.


```python
df.columns
```




    Index(['Description', 'Basic Unit', 'Beginning Balance ', 'Quantity Received ',
           'Quantity Dispensed ', 'Losses and Adjustments',
           'Ending Balance (Physical Count)'],
          dtype='object')




```python
for col in df.columns[2:]:
    df[[not (isinstance(value, int) or isinstance(value, float)) for value in df[col]]]
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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Tab Quinine</td>
      <td>100</td>
      <td>41oo</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3846</th>
      <td>Susp. Erythromycin 125mg/5ml</td>
      <td>Bottle</td>
      <td>o</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>490</th>
      <td>Tabs Ibuprofen 200mg</td>
      <td>1000's</td>
      <td>0</td>
      <td>i</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>491</th>
      <td>Tabs Ibuprofen 200mg</td>
      <td>1000's</td>
      <td>0</td>
      <td>I000</td>
      <td>0</td>
      <td>0</td>
      <td>i000</td>
    </tr>
    <tr>
      <th>605</th>
      <td>Tabs Paracetamol 500mg</td>
      <td>1000's</td>
      <td>0</td>
      <td>s</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>835</th>
      <td>Caps Amoxicillin 500mg</td>
      <td>100's</td>
      <td>0</td>
      <td>20oo</td>
      <td>0</td>
      <td>0</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>1084</th>
      <td>inj Chlorpheniramine 10mg/amp</td>
      <td>100's</td>
      <td>0</td>
      <td>0.S</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3812</th>
      <td>Susp. Co-trimoxazole 240mg/5ml</td>
      <td>Bottle</td>
      <td>30</td>
      <td>O</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4286</th>
      <td>syr Ferric amonium citrate 400mg/5ml</td>
      <td>Bottle</td>
      <td>0</td>
      <td>o</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Surgical gloves</td>
      <td>100</td>
      <td>2</td>
      <td>0</td>
      <td>i</td>
      <td>0</td>
      <td>i</td>
    </tr>
    <tr>
      <th>1781</th>
      <td>Tab. Loratidine</td>
      <td>100's</td>
      <td>2000</td>
      <td>0</td>
      <td>o</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3352</th>
      <td>Hydrogen peroxide</td>
      <td>Bottle</td>
      <td>12</td>
      <td>0</td>
      <td>O</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3354</th>
      <td>lotion Benzylbenzoate lotion</td>
      <td>Bottle</td>
      <td>0</td>
      <td>0</td>
      <td>o</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2950</th>
      <td>Inj Aminophyline</td>
      <td>50's</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>O</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Surgical gloves</td>
      <td>100</td>
      <td>2</td>
      <td>0</td>
      <td>i</td>
      <td>0</td>
      <td>i</td>
    </tr>
    <tr>
      <th>491</th>
      <td>Tabs Ibuprofen 200mg</td>
      <td>1000's</td>
      <td>0</td>
      <td>I000</td>
      <td>0</td>
      <td>0</td>
      <td>i000</td>
    </tr>
    <tr>
      <th>2508</th>
      <td>Tab. Metformin + Glimepride</td>
      <td>30's</td>
      <td>1300</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>O</td>
    </tr>
    <tr>
      <th>5062</th>
      <td>Darrows solution 1/2 strength 500ml</td>
      <td>Pcs</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>o</td>
    </tr>
  </tbody>
</table>
</div>



The above code is looping through the numerical columns and identifying rows within the columns that contains characters that cannot be converted to float/integers. We see errors present in all five numerical columns within the dataset.


```python
alpha_num_mapping = { 'i': '1',
                        'I': '1',
                        'l': '1',
                        'L': '1',
                        'o': '0',
                        'O': '0',
                        's': '5',
                        'S': '5'}
```

Based on the errors we have found, the above mapping is created based on the number I assume the letter is supposed to represent.


```python
for col in df.columns[2:]:
    for old, new in alpha_num_mapping.items():
        df[col] = df[col].replace(old, new, regex=True)
    df[[not (isinstance(value, int) or isinstance(value, float)) for value in df[col]]]
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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>Tab Quinine</td>
      <td>100</td>
      <td>4100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3846</th>
      <td>Susp. Erythromycin 125mg/5ml</td>
      <td>Bottle</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>490</th>
      <td>Tabs Ibuprofen 200mg</td>
      <td>1000's</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>491</th>
      <td>Tabs Ibuprofen 200mg</td>
      <td>1000's</td>
      <td>0</td>
      <td>1000</td>
      <td>0</td>
      <td>0</td>
      <td>i000</td>
    </tr>
    <tr>
      <th>605</th>
      <td>Tabs Paracetamol 500mg</td>
      <td>1000's</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>835</th>
      <td>Caps Amoxicillin 500mg</td>
      <td>100's</td>
      <td>0</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>2000</td>
    </tr>
    <tr>
      <th>1084</th>
      <td>inj Chlorpheniramine 10mg/amp</td>
      <td>100's</td>
      <td>0</td>
      <td>0.5</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3812</th>
      <td>Susp. Co-trimoxazole 240mg/5ml</td>
      <td>Bottle</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4286</th>
      <td>syr Ferric amonium citrate 400mg/5ml</td>
      <td>Bottle</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Surgical gloves</td>
      <td>100</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>i</td>
    </tr>
    <tr>
      <th>1781</th>
      <td>Tab. Loratidine</td>
      <td>100's</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3352</th>
      <td>Hydrogen peroxide</td>
      <td>Bottle</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>3354</th>
      <td>lotion Benzylbenzoate lotion</td>
      <td>Bottle</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2950</th>
      <td>Inj Aminophyline</td>
      <td>50's</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
    </tr>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9</th>
      <td>Surgical gloves</td>
      <td>100</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>491</th>
      <td>Tabs Ibuprofen 200mg</td>
      <td>1000's</td>
      <td>0</td>
      <td>1000</td>
      <td>0</td>
      <td>0</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>2508</th>
      <td>Tab. Metformin + Glimepride</td>
      <td>30's</td>
      <td>1300</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5062</th>
      <td>Darrows solution 1/2 strength 500ml</td>
      <td>Pcs</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We see that the errors were successfully replaced with their corresponding numbers.


```python
for col in df.columns[2:]:
    df[col] = df[col].astype(float)
    df[[not (isinstance(value, int) or isinstance(value, float)) for value in df[col]]]
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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>






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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5846 entries, 0 to 5845
    Data columns (total 7 columns):
     #   Column                           Non-Null Count  Dtype  
    ---  ------                           --------------  -----  
     0   Description                      5846 non-null   object 
     1   Basic Unit                       5846 non-null   object 
     2   Beginning Balance                5846 non-null   float64
     3   Quantity Received                5846 non-null   float64
     4   Quantity Dispensed               5846 non-null   float64
     5   Losses and Adjustments           5846 non-null   float64
     6   Ending Balance (Physical Count)  5846 non-null   float64
    dtypes: float64(5), object(2)
    memory usage: 319.8+ KB
    

After casting the columns to a float datatype, we see that the same code no longer detects any remaining errors. Additionally, there are no null values in the dataframe and we can assume that the data is now cleaned.


```python
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
      <th>Description</th>
      <th>Basic Unit</th>
      <th>Beginning Balance</th>
      <th>Quantity Received</th>
      <th>Quantity Dispensed</th>
      <th>Losses and Adjustments</th>
      <th>Ending Balance (Physical Count)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Tab. Cefuroxime 250mg</td>
      <td>10</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tab. Cefuroxime 250mg</td>
      <td>10</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tab. Cefuroxime 250mg</td>
      <td>10</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Cap Ampicillin</td>
      <td>100</td>
      <td>350.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cap Ampicillin</td>
      <td>100</td>
      <td>350.0</td>
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
    </tr>
    <tr>
      <th>5841</th>
      <td>Inj.Vitamin B Complex</td>
      <td>Vial</td>
      <td>145.0</td>
      <td>30.0</td>
      <td>4350.0</td>
      <td>0.0</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>5842</th>
      <td>Insulin (Actrapid)</td>
      <td>Vial</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5843</th>
      <td>Insulin (Insulutard)</td>
      <td>Vial</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5844</th>
      <td>Inj Amoxicillin 500mg</td>
      <td>vials</td>
      <td>54.0</td>
      <td>100.0</td>
      <td>79.0</td>
      <td>0.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>5845</th>
      <td>Inj Amoxy/clav 1.2</td>
      <td>vials</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5846 rows × 7 columns</p>
</div>



![cleaned_data_7](/images/data_cleaning/data_clean_7.PNG)

<a id='4'></a>
## 4. Invoices with Merged Categories and Merged Amounts
In the final dataset each row is a single transaction and multiple items from the transaction captured within a single column. Our end goal is to convert the dataset into a long format where each item in the transaction will be captured within its own row.


```python
data8 = pd.read_excel('8.-Invoices-with-Merged-Categories-and-Merged-Amounts.xlsx')
df = data8.copy()
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
      <th>Order ID</th>
      <th>Category</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA-2011-167199</td>
      <td>Binders | Art | Phones | Fasteners | Paper</td>
      <td>609.98 | 5.48 | 391.98 | 755.96 | 31.12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CA-2011-149020</td>
      <td>Office Supplies | Furniture</td>
      <td>2.98 | 51.94</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CA-2011-131905</td>
      <td>Office Supplies | Technology | Technology</td>
      <td>7.2 | 42.0186 | 42.035</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-127614</td>
      <td>Accessories | Tables | Binders</td>
      <td>234.45 | 1256.22 | 17.46</td>
    </tr>
  </tbody>
</table>
</div>



![dirty_data_8](/images/data_cleaning/data_dirty_8.PNG)
![cleaned_data_8](/images/data_cleaning/data_clean_8.PNG)

As seen above, the original "dirty" data contains multiple items within "Category" and their corresponding prices within "Amount". The target end state is to have each of this item on its own row within the dataset.


```python
cleaned_df = df.copy()
```


```python
cleaned_df.Category = cleaned_df.Category.str.split('|')
cleaned_df.Amount = cleaned_df.Amount.str.split('|')
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
      <th>Category</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA-2011-167199</td>
      <td>[Binders ,  Art ,  Phones ,  Fasteners ,  Paper]</td>
      <td>[609.98 ,  5.48 ,  391.98 ,  755.96 ,  31.12]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CA-2011-149020</td>
      <td>[Office Supplies ,  Furniture]</td>
      <td>[2.98 ,  51.94]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CA-2011-131905</td>
      <td>[Office Supplies ,  Technology ,  Technology]</td>
      <td>[7.2 ,  42.0186 ,  42.035]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-127614</td>
      <td>[Accessories ,  Tables ,  Binders]</td>
      <td>[234.45 ,  1256.22 ,  17.46]</td>
    </tr>
  </tbody>
</table>
</div>



First we split the two columns by the delimiter | to create a Series of lists each.


```python
cleaned_df = cleaned_df.explode(['Category','Amount']).reset_index(drop=True)
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
      <th>Category</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CA-2011-167199</td>
      <td>Binders</td>
      <td>609.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CA-2011-167199</td>
      <td>Art</td>
      <td>5.48</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CA-2011-167199</td>
      <td>Phones</td>
      <td>391.98</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CA-2011-167199</td>
      <td>Fasteners</td>
      <td>755.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CA-2011-167199</td>
      <td>Paper</td>
      <td>31.12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CA-2011-149020</td>
      <td>Office Supplies</td>
      <td>2.98</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CA-2011-149020</td>
      <td>Furniture</td>
      <td>51.94</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CA-2011-131905</td>
      <td>Office Supplies</td>
      <td>7.2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CA-2011-131905</td>
      <td>Technology</td>
      <td>42.0186</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CA-2011-131905</td>
      <td>Technology</td>
      <td>42.035</td>
    </tr>
    <tr>
      <th>10</th>
      <td>CA-2011-127614</td>
      <td>Accessories</td>
      <td>234.45</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CA-2011-127614</td>
      <td>Tables</td>
      <td>1256.22</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CA-2011-127614</td>
      <td>Binders</td>
      <td>17.46</td>
    </tr>
  </tbody>
</table>
</div>



![cleaned_data_8](/images/data_cleaning/data_clean_8.PNG)

Using Pandas we are able to explode the columns and achieve our final state without additional code. If we are doing this without pd.explode() it will require some zipping of the lists and reassignments to achieve the same result, which will be harder to implement and will take a longer execution time.

With this we have completed cleaning the 8 different datasets, encountering different ways that data can be "dirty" and how we can tackle each of these different situations. In real world scenarios usually data may come in a combination of being poorly structured and incorrect/incomplete entries, which may significantly increase the difficulty of the cleaning process and requires the analyst to combine multiple techniques to achieve the target end state.
