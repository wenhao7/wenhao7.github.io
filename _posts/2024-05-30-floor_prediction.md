---
layout: post
title:  "Kaggle Playground Series S4E5 - Regression with a Flood Prediction Dataset"
date:   2024-05-30 22:31:09 +0800
category: [data_analysis, visualization, machine_learning]
tag: [numpy, pandas, seaborn, matplotlib, scikit-learn, classification, kaggle, competition]
summary: "In this notebook we will be working on the following Kaggle Challenge on a flood detection problem where the goal is to predict the probability of a region flooding based on various factors."
image: /images/banners/flood_prediction.jpg
---

## Contents
1. [Introduction](#1)
2. [EDA](#2)
3. [Feature Engineering](#3)
4. [Modelling](#4)
5. [Prediction](#5)

<a id='1'></a>
## 1. Introduction
In this notebook we will be working on the following [Kaggle Challenge](https://www.kaggle.com/competitions/playground-series-s4e5/overview) on a flood detection problem where the goal is to predict the probability of a region flooding based on various factors.

<a id='2'></a>
## 2. EDA


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor

from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, make_scorer


import warnings
warnings.filterwarnings('ignore')
```


```python
# Import datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train
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
      <th>id</th>
      <th>MonsoonIntensity</th>
      <th>TopographyDrainage</th>
      <th>RiverManagement</th>
      <th>Deforestation</th>
      <th>Urbanization</th>
      <th>ClimateChange</th>
      <th>DamsQuality</th>
      <th>Siltation</th>
      <th>AgriculturalPractices</th>
      <th>...</th>
      <th>DrainageSystems</th>
      <th>CoastalVulnerability</th>
      <th>Landslides</th>
      <th>Watersheds</th>
      <th>DeterioratingInfrastructure</th>
      <th>PopulationScore</th>
      <th>WetlandLoss</th>
      <th>InadequatePlanning</th>
      <th>PoliticalFactors</th>
      <th>FloodProbability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>5</td>
      <td>8</td>
      <td>5</td>
      <td>8</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>7</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
      <td>0.445</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6</td>
      <td>7</td>
      <td>4</td>
      <td>4</td>
      <td>8</td>
      <td>8</td>
      <td>3</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>0.450</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>6</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>...</td>
      <td>7</td>
      <td>3</td>
      <td>7</td>
      <td>5</td>
      <td>6</td>
      <td>8</td>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>0.530</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>5</td>
      <td>4</td>
      <td>8</td>
      <td>4</td>
      <td>7</td>
      <td>6</td>
      <td>...</td>
      <td>2</td>
      <td>4</td>
      <td>7</td>
      <td>4</td>
      <td>4</td>
      <td>6</td>
      <td>5</td>
      <td>7</td>
      <td>5</td>
      <td>0.535</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>3</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>6</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>0.415</td>
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
      <th>1117952</th>
      <td>1117952</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>10</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>10</td>
      <td>...</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>4</td>
      <td>0.495</td>
    </tr>
    <tr>
      <th>1117953</th>
      <td>1117953</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>9</td>
      <td>5</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>9</td>
      <td>4</td>
      <td>4</td>
      <td>3</td>
      <td>7</td>
      <td>4</td>
      <td>9</td>
      <td>4</td>
      <td>5</td>
      <td>0.480</td>
    </tr>
    <tr>
      <th>1117954</th>
      <td>1117954</td>
      <td>7</td>
      <td>3</td>
      <td>9</td>
      <td>4</td>
      <td>6</td>
      <td>5</td>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>0.485</td>
    </tr>
    <tr>
      <th>1117955</th>
      <td>1117955</td>
      <td>7</td>
      <td>3</td>
      <td>3</td>
      <td>7</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>...</td>
      <td>6</td>
      <td>8</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>7</td>
      <td>6</td>
      <td>4</td>
      <td>0.495</td>
    </tr>
    <tr>
      <th>1117956</th>
      <td>1117956</td>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>9</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>8</td>
      <td>4</td>
      <td>...</td>
      <td>4</td>
      <td>8</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
      <td>7</td>
      <td>8</td>
      <td>0.560</td>
    </tr>
  </tbody>
</table>
<p>1117957 rows × 22 columns</p>
</div>




```python
df_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1117957 entries, 0 to 1117956
    Data columns (total 22 columns):
     #   Column                           Non-Null Count    Dtype  
    ---  ------                           --------------    -----  
     0   id                               1117957 non-null  int64  
     1   MonsoonIntensity                 1117957 non-null  int64  
     2   TopographyDrainage               1117957 non-null  int64  
     3   RiverManagement                  1117957 non-null  int64  
     4   Deforestation                    1117957 non-null  int64  
     5   Urbanization                     1117957 non-null  int64  
     6   ClimateChange                    1117957 non-null  int64  
     7   DamsQuality                      1117957 non-null  int64  
     8   Siltation                        1117957 non-null  int64  
     9   AgriculturalPractices            1117957 non-null  int64  
     10  Encroachments                    1117957 non-null  int64  
     11  IneffectiveDisasterPreparedness  1117957 non-null  int64  
     12  DrainageSystems                  1117957 non-null  int64  
     13  CoastalVulnerability             1117957 non-null  int64  
     14  Landslides                       1117957 non-null  int64  
     15  Watersheds                       1117957 non-null  int64  
     16  DeterioratingInfrastructure      1117957 non-null  int64  
     17  PopulationScore                  1117957 non-null  int64  
     18  WetlandLoss                      1117957 non-null  int64  
     19  InadequatePlanning               1117957 non-null  int64  
     20  PoliticalFactors                 1117957 non-null  int64  
     21  FloodProbability                 1117957 non-null  float64
    dtypes: float64(1), int64(21)
    memory usage: 187.6 MB
    


```python
df_train.duplicated().sum()
```




    0



We see that all columns are in a numeric data type with no null values or duplicated entries, suggesting that the data is clean with no risk of hidden strings that does not conform to the numeric nature of the provided features.


```python
df_train.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>1117957.0</td>
      <td>558978.000000</td>
      <td>322726.531782</td>
      <td>0.000</td>
      <td>279489.00</td>
      <td>558978.000</td>
      <td>838467.00</td>
      <td>1117956.000</td>
    </tr>
    <tr>
      <th>MonsoonIntensity</th>
      <td>1117957.0</td>
      <td>4.921450</td>
      <td>2.056387</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>TopographyDrainage</th>
      <td>1117957.0</td>
      <td>4.926671</td>
      <td>2.093879</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>18.000</td>
    </tr>
    <tr>
      <th>RiverManagement</th>
      <td>1117957.0</td>
      <td>4.955322</td>
      <td>2.072186</td>
      <td>0.000</td>
      <td>4.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>Deforestation</th>
      <td>1117957.0</td>
      <td>4.942240</td>
      <td>2.051689</td>
      <td>0.000</td>
      <td>4.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>17.000</td>
    </tr>
    <tr>
      <th>Urbanization</th>
      <td>1117957.0</td>
      <td>4.942517</td>
      <td>2.083391</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>17.000</td>
    </tr>
    <tr>
      <th>ClimateChange</th>
      <td>1117957.0</td>
      <td>4.934093</td>
      <td>2.057742</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>17.000</td>
    </tr>
    <tr>
      <th>DamsQuality</th>
      <td>1117957.0</td>
      <td>4.955878</td>
      <td>2.083063</td>
      <td>0.000</td>
      <td>4.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>Siltation</th>
      <td>1117957.0</td>
      <td>4.927791</td>
      <td>2.065992</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>AgriculturalPractices</th>
      <td>1117957.0</td>
      <td>4.942619</td>
      <td>2.068545</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>Encroachments</th>
      <td>1117957.0</td>
      <td>4.949230</td>
      <td>2.083324</td>
      <td>0.000</td>
      <td>4.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>18.000</td>
    </tr>
    <tr>
      <th>IneffectiveDisasterPreparedness</th>
      <td>1117957.0</td>
      <td>4.945239</td>
      <td>2.078141</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>DrainageSystems</th>
      <td>1117957.0</td>
      <td>4.946893</td>
      <td>2.072333</td>
      <td>0.000</td>
      <td>4.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>17.000</td>
    </tr>
    <tr>
      <th>CoastalVulnerability</th>
      <td>1117957.0</td>
      <td>4.953999</td>
      <td>2.088899</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>17.000</td>
    </tr>
    <tr>
      <th>Landslides</th>
      <td>1117957.0</td>
      <td>4.931376</td>
      <td>2.078287</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>Watersheds</th>
      <td>1117957.0</td>
      <td>4.929032</td>
      <td>2.082395</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>DeterioratingInfrastructure</th>
      <td>1117957.0</td>
      <td>4.925907</td>
      <td>2.064813</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>17.000</td>
    </tr>
    <tr>
      <th>PopulationScore</th>
      <td>1117957.0</td>
      <td>4.927520</td>
      <td>2.074176</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>18.000</td>
    </tr>
    <tr>
      <th>WetlandLoss</th>
      <td>1117957.0</td>
      <td>4.950859</td>
      <td>2.068696</td>
      <td>0.000</td>
      <td>4.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>19.000</td>
    </tr>
    <tr>
      <th>InadequatePlanning</th>
      <td>1117957.0</td>
      <td>4.940587</td>
      <td>2.081123</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>PoliticalFactors</th>
      <td>1117957.0</td>
      <td>4.939004</td>
      <td>2.090350</td>
      <td>0.000</td>
      <td>3.00</td>
      <td>5.000</td>
      <td>6.00</td>
      <td>16.000</td>
    </tr>
    <tr>
      <th>FloodProbability</th>
      <td>1117957.0</td>
      <td>0.504480</td>
      <td>0.051026</td>
      <td>0.285</td>
      <td>0.47</td>
      <td>0.505</td>
      <td>0.54</td>
      <td>0.725</td>
    </tr>
  </tbody>
</table>
</div>



Immediately we see something curious about this dataset. The summary statistics suggests that all feature columns have very similar distributions, likely due to the synthetic nature of the dataset and how it was generated. Hence we will not be attempting to apply any real-world flooding domain specific knowledge to guide us through the challenge.


```python
# Drop useless id column
df_train = df_train.drop('id', axis = 1)
```


```python
sns.heatmap(df_train.corr())
```




    <Axes: >




    
![png](images/flood_prediction/output_9_1.png)
    


The above heatmap shows that all feature columns have insignificant pairwise linear correlations, while all feature columns have some linear correlation with the target column.


```python
df_train.corr()['FloodProbability']
```




    MonsoonIntensity                   0.189098
    TopographyDrainage                 0.187635
    RiverManagement                    0.187131
    Deforestation                      0.184001
    Urbanization                       0.180861
    ClimateChange                      0.184761
    DamsQuality                        0.187996
    Siltation                          0.186789
    AgriculturalPractices              0.183366
    Encroachments                      0.178841
    IneffectiveDisasterPreparedness    0.183109
    DrainageSystems                    0.179305
    CoastalVulnerability               0.177774
    Landslides                         0.185346
    Watersheds                         0.181907
    DeterioratingInfrastructure        0.190007
    PopulationScore                    0.185890
    WetlandLoss                        0.183396
    InadequatePlanning                 0.180968
    PoliticalFactors                   0.182417
    FloodProbability                   1.000000
    Name: FloodProbability, dtype: float64



Similar to what we saw in our summary statistics previously, even the pairwise correlations between each feature and the target are very similar in value.


```python
# Plotting all the distributions
fig, ax = plt.subplots(5, 4, figsize=(16,16))

for col, a in zip(df_train.columns[:-1], ax.reshape(-1)):
    sns.barplot(pd.DataFrame(df_train[col].value_counts()).reset_index(), x=col, y='count', ax = a, color='blue')
    sns.barplot(pd.DataFrame(df_test[col].value_counts()).reset_index(), x=col, y='count', ax = a, color='green')
plt.tight_layout()
plt.show()
```


    
![png](images/flood_prediction/output_13_0.png)
    


After plotting the distributions of the features we can clearly see that all of them have the same distribution


```python
pca = PCA()
pca.fit(df_train.iloc[:,:-1])
pca_df = pd.DataFrame({'Explained Variance':pca.explained_variance_ratio_*100, 'Cumulative Explained Variance':np.cumsum(pca.explained_variance_ratio_)*100})
pca_df['Principal Component'] = list(range(len(pca_df)))
pca_df.head()
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
      <th>Explained Variance</th>
      <th>Cumulative Explained Variance</th>
      <th>Principal Component</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.154727</td>
      <td>5.154727</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.151653</td>
      <td>10.306380</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.131702</td>
      <td>15.438082</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.107014</td>
      <td>20.545097</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.102997</td>
      <td>25.648093</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(pca_df, x='Principal Component', y='Explained Variance', ax=ax, color='green')
sns.lineplot(pca_df, x='Principal Component', y='Cumulative Explained Variance', ax=ax, marker='o', color='blue')
_ = ax.bar_label(ax.containers[0],fmt='%.2f%%', fontsize=9)
for cev, feature in zip(pca_df['Cumulative Explained Variance'], pca_df['Principal Component']):
    ax.annotate(str(round(cev,2)) + '%', (feature-1, cev+2))
plt.title('PCA Explained Variance')
```




    Text(0.5, 1.0, 'PCA Explained Variance')




    
![png](images/flood_prediction/output_16_1.png)
    


From PCA we that the principal components all contribute to the explanation of variance within the dataset.


```python
sns.histplot(df_train['FloodProbability'])
plt.title('Distribution of Target Flood Probabilities')
```




    Text(0.5, 1.0, 'Distribution of Target Flood Probabilities')




    
![png](images/flood_prediction/output_18_1.png)
    


Our target appears to have a normal distribution.

<a id='3'></a>
## 3. Feature Engineering
In this section we will be creating new features that may be helpful in tackling this challenge. We will a bunch of new features by computing some statistics for each sample.


```python
def create_new_features(data, cols):
    df = data.copy()
    df['sum'] = df[cols].sum(axis=1)
    df['mean'] = df[cols].mean(axis=1)
    df['median'] = df[cols].median(axis=1)
    df['max'] = df[cols].max(axis=1)
    df['min'] = df[cols].min(axis=1)
    df['std'] = df[cols].std(axis=1)
    df['cov'] = df['std']/df['mean']
    df['p25'] = df[cols].quantile(0.25, axis=1)
    df['p75'] = df[cols].quantile(0.75, axis=1)
    df['range'] = df['max'] - df['min']
    return df
```


```python
df_train_new = create_new_features(df_train, df_train.columns[:-1])
df_train_new_only = df_train_new.drop(df_train.columns[:-1], axis = 1)
```


```python
df_train_new_only
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
      <th>FloodProbability</th>
      <th>sum</th>
      <th>mean</th>
      <th>median</th>
      <th>max</th>
      <th>min</th>
      <th>std</th>
      <th>cov</th>
      <th>p25</th>
      <th>p75</th>
      <th>range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.445</td>
      <td>94</td>
      <td>4.70</td>
      <td>4.5</td>
      <td>8</td>
      <td>2</td>
      <td>1.750188</td>
      <td>0.372380</td>
      <td>3.00</td>
      <td>5.25</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.450</td>
      <td>94</td>
      <td>4.70</td>
      <td>4.0</td>
      <td>9</td>
      <td>0</td>
      <td>2.296450</td>
      <td>0.488606</td>
      <td>3.00</td>
      <td>6.25</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.530</td>
      <td>99</td>
      <td>4.95</td>
      <td>5.0</td>
      <td>8</td>
      <td>1</td>
      <td>1.932411</td>
      <td>0.390386</td>
      <td>3.00</td>
      <td>6.25</td>
      <td>7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.535</td>
      <td>104</td>
      <td>5.20</td>
      <td>5.0</td>
      <td>8</td>
      <td>2</td>
      <td>1.641565</td>
      <td>0.315686</td>
      <td>4.00</td>
      <td>6.25</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.415</td>
      <td>72</td>
      <td>3.60</td>
      <td>3.0</td>
      <td>6</td>
      <td>1</td>
      <td>1.500877</td>
      <td>0.416910</td>
      <td>2.75</td>
      <td>5.00</td>
      <td>5</td>
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
    </tr>
    <tr>
      <th>1117952</th>
      <td>0.495</td>
      <td>99</td>
      <td>4.95</td>
      <td>4.0</td>
      <td>10</td>
      <td>1</td>
      <td>2.543826</td>
      <td>0.513904</td>
      <td>3.00</td>
      <td>7.00</td>
      <td>9</td>
    </tr>
    <tr>
      <th>1117953</th>
      <td>0.480</td>
      <td>96</td>
      <td>4.80</td>
      <td>4.0</td>
      <td>9</td>
      <td>1</td>
      <td>2.419221</td>
      <td>0.504004</td>
      <td>3.00</td>
      <td>5.50</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1117954</th>
      <td>0.485</td>
      <td>98</td>
      <td>4.90</td>
      <td>5.0</td>
      <td>9</td>
      <td>1</td>
      <td>1.970840</td>
      <td>0.402212</td>
      <td>4.00</td>
      <td>5.25</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1117955</th>
      <td>0.495</td>
      <td>99</td>
      <td>4.95</td>
      <td>5.0</td>
      <td>8</td>
      <td>2</td>
      <td>1.700619</td>
      <td>0.343559</td>
      <td>3.75</td>
      <td>6.00</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1117956</th>
      <td>0.560</td>
      <td>110</td>
      <td>5.50</td>
      <td>5.0</td>
      <td>9</td>
      <td>1</td>
      <td>2.013115</td>
      <td>0.366021</td>
      <td>4.75</td>
      <td>7.00</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>1117957 rows × 11 columns</p>
</div>




```python
sns.heatmap(df_train_new_only.corr(), annot=True)
```




    <Axes: >




    
![png](images/flood_prediction/output_23_1.png)
    


Looking at pairwise linear correlation of the new features and the target, sum and mean values of all the features per sample have a high correlation of 0.92. The remaining features have varying levels of correlation with the target, with only standard deviation and range having a lower correlation coefficient than the original features.


```python
fig, ax = plt.subplots(6, 5, figsize=(16,16))

for col, a in zip(df_train_new.drop('FloodProbability', axis=1).columns, ax.reshape(-1)):
    sns.scatterplot(df_train_new[[col,'FloodProbability']], x=col, y='FloodProbability', ax = a, color='blue')
plt.tight_layout()
plt.show()
```


    
![png](images/flood_prediction/output_25_0.png)
    


Looking at scatterplots of all the features we have vs FloodProbability, we are able to see some of the relationships visually. In particular, positive linear correlation is very visible on sum, mean, median, max, ps25, and ps75.

<a id='4'></a>
## 4. Modelling
For modelling we will be comparing linear regression with regularization, xgboost, and lightgbm algorithms. The model parameters will be tuned and additional voting and stacking estimators will be included in our final comparison to find our best performing estimator.


```python
# Instatiate models and define parameters to gridsearch
random_state = 47
linreg = ElasticNet(random_state = random_state)

linreg_params = {'alpha':[0.1, 1, 10],
                    'l1_ratio':[0.25, 0.5, 0.75]}


xgb = XGBRegressor(random_state = random_state,
                    objective='reg:linear',
                    metric='r2',
                    device = 'cuda', error_score='raise')

xgb_params = {'learning_rate': [0.03, .07],
              'max_depth': [6, 9],
              'min_child_weight': [1,10],
              'colsample_bytree': [0.5,1]
             }


lgb = LGBMRegressor(random_state = random_state,
                        metric='r2',
                        verbosity=-1)

lgb_params = {'num_leaves':[10,31],
                'learning_rate':[ 0.01, 0.1],  
                'colsample_bytree': [0.5, 1],
                'reg_alpha': [0, 0.05], 
                'reg_lambda': [0, 0.05]
             } 


rgs = [
    ('Linear Regression', linreg, linreg_params),
    ('XGBoost Regressior', xgb, xgb_params),
    ('LGBM Regressor', lgb, lgb_params)
]
scorer = {
    'r2_score': make_scorer(r2_score),
    'mae_score': make_scorer(mean_squared_error),
}
```


```python
X_train = df_train_new.drop('FloodProbability', axis=1)
y_train = df_train_new['FloodProbability']
X_train_new = df_train_new_only.drop('FloodProbability', axis=1)
print(X_train.shape)
print(X_train_new.shape)
```

    (1117957, 30)
    (1117957, 10)
    

We will evaluate three different datasets
1. All features, scaled
2. All features, not scaled
3. Only new features, scaled


```python
# Three different pipelines
results = []

for rg_name, rg, rg_params in rgs:
    gs = GridSearchCV(estimator=rg,
                     param_grid=rg_params,
                     scoring=scorer,
                     refit='r2_score',
                     verbose=2,
                     error_score='raise')
    
    # Original + New Features, Scaled
    pipeline_a = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('estimator', gs),
    ])
    pipeline_a.fit(X_train, y_train)
    result = ['All features scaled', rg_name, gs.best_params_, gs.best_score_, gs.cv_results_['mean_test_r2_score'][gs.best_index_], gs.cv_results_['mean_test_mae_score'][gs.best_index_]]
    results.append(result)

    # Original + New Features, Not scaled
    pipeline_b = Pipeline(steps=[
        ('estimator', gs),
    ])
    pipeline_b.fit(X_train, y_train)
    result = ['All features', rg_name, gs.best_params_, gs.best_score_, gs.cv_results_['mean_test_r2_score'][gs.best_index_], gs.cv_results_['mean_test_mae_score'][gs.best_index_]]
    results.append(result)

    # New Features only, Scaled
    pipeline_c = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('estimator', gs),
    ])
    pipeline_c.fit(X_train_new, y_train)
    result = ['New features scaled', rg_name, gs.best_params_, gs.best_score_, gs.cv_results_['mean_test_r2_score'][gs.best_index_], gs.cv_results_['mean_test_mae_score'][gs.best_index_]]
    results.append(result)

```

    Fitting 5 folds for each of 9 candidates, totalling 45 fits
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.7s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.8s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.8s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.7s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.8s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.4s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.4s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.6s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.4s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.5s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.4s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.4s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.4s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.4s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.4s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.4s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.4s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.4s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.3s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.4s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.4s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.4s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.4s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.4s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.4s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.4s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.4s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.4s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.4s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.3s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.4s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.3s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.3s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.4s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.4s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.4s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.4s
    Fitting 5 folds for each of 9 candidates, totalling 45 fits
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.3s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.3s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.3s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.3s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.3s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.3s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.3s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.3s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.3s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.3s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.3s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.3s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.3s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.3s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.3s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.3s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.3s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.3s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.3s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.3s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.4s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.3s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.3s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.3s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.3s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.3s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.3s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.3s
    Fitting 5 folds for each of 9 candidates, totalling 45 fits
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.2s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.2s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.2s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.2s
    [CV] END ...........................alpha=0.1, l1_ratio=0.25; total time=   0.2s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.0s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.0s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.0s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.1s
    [CV] END ............................alpha=0.1, l1_ratio=0.5; total time=   0.0s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.0s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.0s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.0s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.0s
    [CV] END ...........................alpha=0.1, l1_ratio=0.75; total time=   0.0s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.1s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.1s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.1s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.0s
    [CV] END .............................alpha=1, l1_ratio=0.25; total time=   0.0s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.0s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.1s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.0s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.0s
    [CV] END ..............................alpha=1, l1_ratio=0.5; total time=   0.1s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.0s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.0s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.0s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.1s
    [CV] END .............................alpha=1, l1_ratio=0.75; total time=   0.0s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.0s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.1s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.0s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.0s
    [CV] END ............................alpha=10, l1_ratio=0.25; total time=   0.0s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.1s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.0s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.0s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.0s
    [CV] END .............................alpha=10, l1_ratio=0.5; total time=   0.0s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.0s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.0s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.0s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.0s
    [CV] END ............................alpha=10, l1_ratio=0.75; total time=   0.0s
    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   4.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   4.0s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.7s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.8s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.8s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.5s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.6s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.5s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.6s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.6s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.6s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.6s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.6s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.6s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.3s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.7s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.6s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.6s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.6s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.2s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.2s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   4.8s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   4.9s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.0s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.0s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.0s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   4.9s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   4.9s
    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   4.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   4.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   4.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   4.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   5.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   5.1s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.8s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   4.0s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   4.0s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   4.0s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   4.0s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.4s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.7s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.7s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.6s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   5.5s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.5s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.4s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.1s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   5.2s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   3.3s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   4.7s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   4.7s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   4.9s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   4.7s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   4.8s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   4.7s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   4.7s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   4.8s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   4.6s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   4.6s
    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.1s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=6, min_child_weight=10; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=1; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.3s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.3s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.3s
    [CV] END colsample_bytree=1, learning_rate=0.03, max_depth=9, min_child_weight=10; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=1; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=6, min_child_weight=10; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.0s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.0s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.0s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=1; total time=   3.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.1s
    [CV] END colsample_bytree=1, learning_rate=0.07, max_depth=9, min_child_weight=10; total time=   3.1s
    Fitting 5 folds for each of 32 candidates, totalling 160 fits
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.5s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   3.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   3.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.3s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.6s
    Fitting 5 folds for each of 32 candidates, totalling 160 fits
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   4.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   4.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.9s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   4.5s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   4.1s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   4.0s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   4.0s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   3.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   3.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.5s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.8s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   2.3s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   2.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   3.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   3.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   3.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   3.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.9s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   2.9s
    Fitting 5 folds for each of 32 candidates, totalling 160 fits
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.2s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.3s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   2.1s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=0.5, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.2s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.3s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   2.0s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.6s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   2.4s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.9s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.8s
    [CV] END colsample_bytree=1, learning_rate=0.01, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.7s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.2s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.0s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=10, reg_alpha=0.05, reg_lambda=0.05; total time=   1.1s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.4s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.6s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.5s
    [CV] END colsample_bytree=1, learning_rate=0.1, num_leaves=31, reg_alpha=0.05, reg_lambda=0.05; total time=   1.5s
    


```python
result_df = pd.DataFrame(results, columns=['Pipeline','Model','Parameters','Best_Score','R2_Score','MSE'])
result_df['Name'] = result_df['Model'] + '_' + result_df['Pipeline']

palette = sns.color_palette()
cmap = {}
for d, color in zip(set(result_df['Pipeline']), palette):
    cmap[d] = color
result_df['Color'] = [cmap[d] for d in result_df['Pipeline']]
result_df.head()
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
      <th>Pipeline</th>
      <th>Model</th>
      <th>Parameters</th>
      <th>Best_Score</th>
      <th>R2_Score</th>
      <th>MSE</th>
      <th>Name</th>
      <th>Color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>All features scaled</td>
      <td>Linear Regression</td>
      <td>{'alpha': 0.1, 'l1_ratio': 0.25}</td>
      <td>0.589445</td>
      <td>0.589445</td>
      <td>0.001069</td>
      <td>Linear Regression_All features scaled</td>
      <td>(0.17254901960784313, 0.6274509803921569, 0.17...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>All features</td>
      <td>Linear Regression</td>
      <td>{'alpha': 0.1, 'l1_ratio': 0.25}</td>
      <td>0.841341</td>
      <td>0.841341</td>
      <td>0.000413</td>
      <td>Linear Regression_All features</td>
      <td>(0.12156862745098039, 0.4666666666666667, 0.70...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>New features scaled</td>
      <td>Linear Regression</td>
      <td>{'alpha': 0.1, 'l1_ratio': 0.25}</td>
      <td>0.589445</td>
      <td>0.589445</td>
      <td>0.001069</td>
      <td>Linear Regression_New features scaled</td>
      <td>(1.0, 0.4980392156862745, 0.054901960784313725)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>All features scaled</td>
      <td>XGBoost Regressior</td>
      <td>{'colsample_bytree': 1, 'learning_rate': 0.07,...</td>
      <td>0.869006</td>
      <td>0.869006</td>
      <td>0.000341</td>
      <td>XGBoost Regressior_All features scaled</td>
      <td>(0.17254901960784313, 0.6274509803921569, 0.17...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>All features</td>
      <td>XGBoost Regressior</td>
      <td>{'colsample_bytree': 1, 'learning_rate': 0.07,...</td>
      <td>0.869006</td>
      <td>0.869006</td>
      <td>0.000341</td>
      <td>XGBoost Regressior_All features</td>
      <td>(0.12156862745098039, 0.4666666666666667, 0.70...</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots()
sns.barplot(result_df.sort_values('Best_Score', ascending=False), orient='h', x='Best_Score', y='Name', palette=result_df['Color'].values, ax=ax)
ax.bar_label(ax.containers[0],fmt='%.4f')
ax.set_xlim(min(result_df['Best_Score'])*0.9, max(result_df['Best_Score'])*1.1)
plt.title('GridSearchCV Model Scores')
plt.show()
```


    
![png](images/flood_prediction/output_32_0.png)
    


All features, scaled led to the best performing models, with XGB and LGB having the best performance. We will use these two models in our stacking/voting estimators.

<a id='5'></a>
## 5. Prediction


```python
result_df[(result_df['Model'].isin(['XGBoost Regressior','LGBM Regressor']))&(result_df['Pipeline']=='All features scaled')]['Parameters'].tolist()
```




    [{'colsample_bytree': 1,
      'learning_rate': 0.07,
      'max_depth': 9,
      'min_child_weight': 10},
     {'colsample_bytree': 1,
      'learning_rate': 0.1,
      'num_leaves': 31,
      'reg_alpha': 0.05,
      'reg_lambda': 0}]



We will feed these hyperparameters from our gridsearch into our new pipeline to include stacking/voting in our comparison.


```python
scaler = StandardScaler()
xgb = XGBRegressor(colsample_bytree= 1,
                  learning_rate= 0.07,
                  max_depth= 9,
                  min_child_weight= 10)
lgb = LGBMRegressor(colsample_bytree= 1,
                  learning_rate= 0.1,
                  num_leaves= 31,
                  reg_alpha= 0.05,
                  reg_lambda= 0)
vr = VotingRegressor(estimators=[('xgb',xgb),('lgb',lgb)])
sr = StackingRegressor(estimators=[('xgb',xgb),('lgb',lgb)])

rgs = [
    ('XGBoost Regressor', xgb),
    ('LGBM Regressor', lgb),
    ('Voting Regressor', vr),
    ('Stacking Regressor', sr)
]
```


```python
results = []
for name, rg in rgs:
    pipeline = Pipeline(
        [('scaling', scaler),
        ('estimator', rg)])
    cv = KFold(n_splits=5)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2')
    results.append([name, scores])
```

    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.047648 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1123
    [LightGBM] [Info] Number of data points in the train set: 894365, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504471
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.046979 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 894365, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504463
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.048580 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1118
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504470
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.051768 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1114
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504528
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.018225 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504469
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.051251 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1123
    [LightGBM] [Info] Number of data points in the train set: 894365, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504471
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.049724 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 894365, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504463
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.054012 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1118
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504470
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.048524 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1114
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504528
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.050193 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504469
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.022039 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 1123
    [LightGBM] [Info] Number of data points in the train set: 894365, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504471
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.040536 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504472
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.038065 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1119
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504440
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.039148 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1117
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504478
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.040978 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1121
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504526
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.049509 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1119
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504439
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.051581 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 894365, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504463
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.042138 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1117
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504487
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.039552 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1116
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504415
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.039807 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1112
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504468
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.040954 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504516
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.039980 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1110
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504429
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.049403 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1118
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504470
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.036748 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1118
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504496
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.041109 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1121
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504437
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.041865 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1125
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504455
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.047561 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1121
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504525
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.046104 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1122
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504438
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.048991 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1114
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504528
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.043192 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1120
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504568
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.044097 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1120
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504509
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.038326 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1118
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504504
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.045451 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1119
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504548
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.038615 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1120
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504510
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.045754 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504469
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.037155 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1118
    [LightGBM] [Info] Number of data points in the train set: 715492, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504495
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.038596 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1119
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504436
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.034958 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1125
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504431
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.037054 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1119
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504474
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.036766 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1120
    [LightGBM] [Info] Number of data points in the train set: 715493, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504511
    


```python
final_results = pd.DataFrame(results, columns=['Model','R2_Score'])
final_results['Mean_Score'] = final_results['R2_Score'].apply(np.mean)
```


```python
final_results
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
      <th>Model</th>
      <th>R2_Score</th>
      <th>Mean_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>XGBoost Regressor</td>
      <td>[0.869121124554708, 0.8676401043047227, 0.8691...</td>
      <td>0.869006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LGBM Regressor</td>
      <td>[0.8689425408537339, 0.8673734831606648, 0.868...</td>
      <td>0.868775</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Voting Regressor</td>
      <td>[0.8691877330536725, 0.8676634426243792, 0.869...</td>
      <td>0.869045</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Stacking Regressor</td>
      <td>[0.8692026715714507, 0.8676879048270891, 0.869...</td>
      <td>0.869065</td>
    </tr>
  </tbody>
</table>
</div>



Performances of all four estimators are similar, but from the CV results we see that stacking led to the best performance. Hence we will submitting our predictions for this challenge using our stacking pipeline.


```python
# Target format for submission
pd.read_csv('sample_submission.csv')
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
      <th>id</th>
      <th>FloodProbability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1117957</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1117958</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1117959</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1117960</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1117961</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>745300</th>
      <td>1863257</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>745301</th>
      <td>1863258</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>745302</th>
      <td>1863259</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>745303</th>
      <td>1863260</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>745304</th>
      <td>1863261</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
<p>745305 rows × 2 columns</p>
</div>




```python
# Final train and test splits
scaler = StandardScaler()
X_train = df_train_new.drop('FloodProbability', axis=1)
y_train = df_train_new['FloodProbability']
X_train = scaler.fit_transform(X_train)

X_test = df_test.drop('id', axis=1)
X_test = create_new_features(X_test, X_test.columns)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test.shape)
```

    (1117957, 30)
    (745305, 30)
    


```python
# Final models
xgb = XGBRegressor(colsample_bytree= 1,
                  learning_rate= 0.07,
                  max_depth= 9,
                  min_child_weight= 10)
lgb = LGBMRegressor(colsample_bytree= 1,
                  learning_rate= 0.1,
                  num_leaves= 31,
                  reg_alpha= 0.05,
                  reg_lambda= 0)
sr = StackingRegressor(estimators=[('xgb',xgb),('lgb',lgb)])
```


```python
sr.fit(X_train, y_train)
y_pred = sr.predict(X_test)
```

    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.025340 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 1120
    [LightGBM] [Info] Number of data points in the train set: 1117957, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504480
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.050135 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1123
    [LightGBM] [Info] Number of data points in the train set: 894365, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504471
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.047327 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 894365, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504463
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.055870 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1118
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504470
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.046390 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1114
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504528
    [LightGBM] [Info] Auto-choosing col-wise multi-threading, the overhead of testing was 0.051994 seconds.
    You can set `force_col_wise=true` to remove the overhead.
    [LightGBM] [Info] Total Bins 1115
    [LightGBM] [Info] Number of data points in the train set: 894366, number of used features: 30
    [LightGBM] [Info] Start training from score 0.504469
    


```python
pred_df = df_test.copy()[['id']]
pred_df['FloodProbability'] = y_pred
pred_df
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
      <th>id</th>
      <th>FloodProbability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1117957</td>
      <td>0.578240</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1117958</td>
      <td>0.456551</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1117959</td>
      <td>0.449741</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1117960</td>
      <td>0.466643</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1117961</td>
      <td>0.466660</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>745300</th>
      <td>1863257</td>
      <td>0.475449</td>
    </tr>
    <tr>
      <th>745301</th>
      <td>1863258</td>
      <td>0.444587</td>
    </tr>
    <tr>
      <th>745302</th>
      <td>1863259</td>
      <td>0.619708</td>
    </tr>
    <tr>
      <th>745303</th>
      <td>1863260</td>
      <td>0.549273</td>
    </tr>
    <tr>
      <th>745304</th>
      <td>1863261</td>
      <td>0.528544</td>
    </tr>
  </tbody>
</table>
<p>745305 rows × 2 columns</p>
</div>




```python
# Export predictions
pred_df.to_csv('submission.csv',index=False)
```

Our evaluation score on the public leaderboard is 0.86902 which at the time of writing places us within the top 10% of this Kaggle playground.
