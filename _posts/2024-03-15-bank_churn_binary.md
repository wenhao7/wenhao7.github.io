---
layout: post
title:  "Bank Churn Exploration and Binary Classification"
date:   2024-03-15 22:31:09 +0800
category: [data_analysis, visualization, machine_learning]
tag: [numpy, pandas, seaborn, matplotlib, scikit-learn, classification]
summary: "In this notebook we will explore a synthetic bank customer churn dataset used in a Kaggle community prediction competition, treating this like a real world problem and avoiding the use of any performance-boosting tricks that is are only specific to this competition dataset (i.e. utilizing data leakages due to the syntheticity of the data."
image: /images/banners/bank.png
---

## Contents
1. [Introduction](#intro)
2. [Data Cleaning](#cleaning)
3. [Data Exploration](#eda)
4. [Feature Engineering](#engineering)
5. [Modelling](#modelling)
6. [Conclusion](#conclusion)
   
***

## 1. Introduction <a id='intro'></a>
In this notebook we will explore a synthetic bank customer churn dataset used in a [Kaggle community prediction competition](https://www.kaggle.com/competitions/copy-of-binary-classification-with-a-bank-churnqq/overview), treating this like a real world problem and avoiding the use of any performance-boosting tricks that is are only specific to this competition dataset (i.e. utilizing data leakages due to the syntheticity of the data)

### Churn - What and why?
Customer churn is a measure of how many customers leave the bank entirely. Customers may leave the bank due to many different reasons, a few common ones are:
    <ol>
        <li>Dissatisfaction with the products offered.</li>
        <li>Competitor offerings are more appealing.</li>
        <li>External factors that make it impossible for the customer to continue using the bank services.</li>
    </ol>

<a id='cleaning'></a>
## 2. Data Cleaning


```python
# Import packages and set styles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, auc, roc_curve, roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

plt.style.use('fivethirtyeight')
sns.set_palette("muted")
```


```python
df = pd.read_csv('train.csv')
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
      <th>id</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>15674932</td>
      <td>Okwudilichukwu</td>
      <td>668</td>
      <td>France</td>
      <td>Male</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>181449.97</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>15749177</td>
      <td>Okwudiliolisa</td>
      <td>627</td>
      <td>France</td>
      <td>Male</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>49503.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>15694510</td>
      <td>Hsueh</td>
      <td>678</td>
      <td>France</td>
      <td>Male</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>184866.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15741417</td>
      <td>Kao</td>
      <td>581</td>
      <td>France</td>
      <td>Male</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>84560.88</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>15766172</td>
      <td>Chiemenam</td>
      <td>716</td>
      <td>Spain</td>
      <td>Male</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15068.83</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (165034, 14)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 165034 entries, 0 to 165033
    Data columns (total 14 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   id               165034 non-null  int64  
     1   CustomerId       165034 non-null  int64  
     2   Surname          165034 non-null  object 
     3   CreditScore      165034 non-null  int64  
     4   Geography        165034 non-null  object 
     5   Gender           165034 non-null  object 
     6   Age              165034 non-null  float64
     7   Tenure           165034 non-null  int64  
     8   Balance          165034 non-null  float64
     9   NumOfProducts    165034 non-null  int64  
     10  HasCrCard        165034 non-null  float64
     11  IsActiveMember   165034 non-null  float64
     12  EstimatedSalary  165034 non-null  float64
     13  Exited           165034 non-null  int64  
    dtypes: float64(5), int64(6), object(3)
    memory usage: 17.6+ MB
    

Column descriptions for the features within the dataset
1. Customer ID: A unique identifier for each customer
2. Surname: The customer's surname or last name
3. Credit Score: A numerical value representing the customer's credit score
4. Geography: The country where the customer resides (France, Spain or Germany)
5. Gender: The customer's gender (Male or Female)
6. Age: The customer's age.
7. Tenure: The number of years the customer has been with the bank
8. Balance: The customer's account balance
9. NumOfProducts: The number of bank products the customer uses (e.g., savings account, credit card)
10. HasCrCard: Whether the customer has a credit card (1 = yes, 0 = no)
11. IsActiveMember: Whether the customer is an active member (1 = yes, 0 = no)
12. EstimatedSalary: The estimated salary of the customer
13. Exited: Whether the customer has churned (1 = yes, 0 = no)


```python
print(df.duplicated().sum())
print(df.isna().sum())
```

    0
    id                 0
    CustomerId         0
    Surname            0
    CreditScore        0
    Geography          0
    Gender             0
    Age                0
    Tenure             0
    Balance            0
    NumOfProducts      0
    HasCrCard          0
    IsActiveMember     0
    EstimatedSalary    0
    Exited             0
    dtype: int64
    

On the surface it appears that there are no duplicates in this dataset, however looking deeper we see that the CustomerId column which is supposed to be a unique identifier for each customer is apparently not so unique after all!


```python
df[df['CustomerId'].duplicated()]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>99</td>
      <td>15673599</td>
      <td>Williamson</td>
      <td>618</td>
      <td>Spain</td>
      <td>Male</td>
      <td>35.0</td>
      <td>5</td>
      <td>133476.09</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>154843.40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>113</th>
      <td>113</td>
      <td>15690958</td>
      <td>Palerma</td>
      <td>594</td>
      <td>France</td>
      <td>Male</td>
      <td>35.0</td>
      <td>2</td>
      <td>185732.59</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>155843.48</td>
      <td>0</td>
    </tr>
    <tr>
      <th>122</th>
      <td>122</td>
      <td>15606887</td>
      <td>Olejuru</td>
      <td>762</td>
      <td>France</td>
      <td>Male</td>
      <td>29.0</td>
      <td>8</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>43075.70</td>
      <td>0</td>
    </tr>
    <tr>
      <th>124</th>
      <td>124</td>
      <td>15741417</td>
      <td>Ts'ui</td>
      <td>706</td>
      <td>France</td>
      <td>Female</td>
      <td>42.0</td>
      <td>8</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>167778.61</td>
      <td>0</td>
    </tr>
    <tr>
      <th>160</th>
      <td>160</td>
      <td>15763612</td>
      <td>Y?an</td>
      <td>712</td>
      <td>France</td>
      <td>Female</td>
      <td>43.0</td>
      <td>4</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117038.96</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>165029</th>
      <td>165029</td>
      <td>15667085</td>
      <td>Meng</td>
      <td>667</td>
      <td>Spain</td>
      <td>Female</td>
      <td>33.0</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>131834.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>165030</th>
      <td>165030</td>
      <td>15665521</td>
      <td>Okechukwu</td>
      <td>792</td>
      <td>France</td>
      <td>Male</td>
      <td>35.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>131834.45</td>
      <td>0</td>
    </tr>
    <tr>
      <th>165031</th>
      <td>165031</td>
      <td>15664752</td>
      <td>Hsia</td>
      <td>565</td>
      <td>France</td>
      <td>Male</td>
      <td>31.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>127429.56</td>
      <td>0</td>
    </tr>
    <tr>
      <th>165032</th>
      <td>165032</td>
      <td>15689614</td>
      <td>Hsiung</td>
      <td>554</td>
      <td>Spain</td>
      <td>Female</td>
      <td>30.0</td>
      <td>7</td>
      <td>161533.00</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>71173.03</td>
      <td>0</td>
    </tr>
    <tr>
      <th>165033</th>
      <td>165033</td>
      <td>15732798</td>
      <td>Ulyanov</td>
      <td>850</td>
      <td>France</td>
      <td>Male</td>
      <td>31.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>61581.79</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>141813 rows × 14 columns</p>
</div>




```python
df.loc[df['CustomerId']==15732798]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>62381</th>
      <td>62381</td>
      <td>15732798</td>
      <td>H?</td>
      <td>733</td>
      <td>France</td>
      <td>Female</td>
      <td>35.0</td>
      <td>6</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>52301.15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83124</th>
      <td>83124</td>
      <td>15732798</td>
      <td>Chukwubuikem</td>
      <td>607</td>
      <td>Germany</td>
      <td>Female</td>
      <td>53.0</td>
      <td>5</td>
      <td>121490.04</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>101039.76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>139366</th>
      <td>139366</td>
      <td>15732798</td>
      <td>Hsueh</td>
      <td>652</td>
      <td>Germany</td>
      <td>Male</td>
      <td>28.0</td>
      <td>1</td>
      <td>171770.43</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>153373.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>156934</th>
      <td>156934</td>
      <td>15732798</td>
      <td>Hsueh</td>
      <td>637</td>
      <td>France</td>
      <td>Male</td>
      <td>32.0</td>
      <td>1</td>
      <td>121520.41</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>77965.49</td>
      <td>0</td>
    </tr>
    <tr>
      <th>165033</th>
      <td>165033</td>
      <td>15732798</td>
      <td>Ulyanov</td>
      <td>850</td>
      <td>France</td>
      <td>Male</td>
      <td>31.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>61581.79</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



We see that there are multiple different customers associated with the above CustomerId. We can tell that they are most likely different individuals carrying different surnames/living in different countries/are of different genders.

To investigate further we shall apply a group by condition to make the condition more specific


```python
df[df[['CustomerId', 'Surname', 'Geography', 'Gender']].duplicated()]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>779</th>
      <td>779</td>
      <td>15718773</td>
      <td>Pisano</td>
      <td>605</td>
      <td>France</td>
      <td>Female</td>
      <td>37.0</td>
      <td>0</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>160129.99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>1132</td>
      <td>15694272</td>
      <td>Nkemakolam</td>
      <td>665</td>
      <td>France</td>
      <td>Male</td>
      <td>38.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>77783.35</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1280</th>
      <td>1280</td>
      <td>15626012</td>
      <td>Obidimkpa</td>
      <td>459</td>
      <td>France</td>
      <td>Male</td>
      <td>48.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>50016.17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1340</th>
      <td>1340</td>
      <td>15589793</td>
      <td>Onwuamaeze</td>
      <td>633</td>
      <td>France</td>
      <td>Male</td>
      <td>53.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>190998.96</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1387</th>
      <td>1387</td>
      <td>15598097</td>
      <td>Johnstone</td>
      <td>651</td>
      <td>France</td>
      <td>Male</td>
      <td>44.0</td>
      <td>9</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>26257.01</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>164974</th>
      <td>164974</td>
      <td>15774882</td>
      <td>Mazzanti</td>
      <td>687</td>
      <td>Germany</td>
      <td>Female</td>
      <td>35.0</td>
      <td>3</td>
      <td>99587.43</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1713.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>164977</th>
      <td>164977</td>
      <td>15704466</td>
      <td>Udokamma</td>
      <td>548</td>
      <td>France</td>
      <td>Female</td>
      <td>34.0</td>
      <td>2</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>195074.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>164982</th>
      <td>164982</td>
      <td>15592999</td>
      <td>Fang</td>
      <td>535</td>
      <td>France</td>
      <td>Female</td>
      <td>42.0</td>
      <td>6</td>
      <td>0.00</td>
      <td>1</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>185660.30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>164983</th>
      <td>164983</td>
      <td>15694192</td>
      <td>Nwankwo</td>
      <td>598</td>
      <td>France</td>
      <td>Female</td>
      <td>38.0</td>
      <td>6</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>173783.38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>165006</th>
      <td>165006</td>
      <td>15627665</td>
      <td>Sung</td>
      <td>614</td>
      <td>France</td>
      <td>Female</td>
      <td>39.0</td>
      <td>4</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>74379.57</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>19154 rows × 14 columns</p>
</div>




```python
df.loc[(df['CustomerId']==15627665)&(df['Surname']=='Sung')&(df['Geography']=='France')]
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16987</th>
      <td>16987</td>
      <td>15627665</td>
      <td>Sung</td>
      <td>614</td>
      <td>France</td>
      <td>Male</td>
      <td>46.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>74379.57</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22713</th>
      <td>22713</td>
      <td>15627665</td>
      <td>Sung</td>
      <td>642</td>
      <td>France</td>
      <td>Female</td>
      <td>60.0</td>
      <td>8</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>74379.57</td>
      <td>0</td>
    </tr>
    <tr>
      <th>165006</th>
      <td>165006</td>
      <td>15627665</td>
      <td>Sung</td>
      <td>614</td>
      <td>France</td>
      <td>Female</td>
      <td>39.0</td>
      <td>4</td>
      <td>0.0</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>74379.57</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Picking CustomerId 15627665 which met the group by condition we see that it is associated with 2 customers that share the same surname, country, and gender. However, looking at the Age we see that they are 21 years apart, with tenures short enough that we cannot say they are the same individual enrolled in the bank's services at different points in her life. 

Looking at the data we have and any additional information/data dictionary from the data source we can conclude that there is not enough information for us to treat similar CustomerIds as a single customer. Hence in this notebook we will proceed as if each row is a unique customer.

In reality if working with banks such an error in the data should not occur due to strict regulations and scrutiny they are subjected to. If it does happen, a thorough investigation should happen and we can seek further clarity on the dataset accordingly.
<a id='eda'></a>
## 3. Exploratory Data Analysis


```python
num_cols = df.select_dtypes(exclude='object').columns
cat_cols = df.select_dtypes('object').columns

for col in cat_cols:
    print("Number of unique categories: ", df[col].nunique())
    print(df[col].value_counts(), "\n---------------------------------\n")
```

    Number of unique categories:  2797
    Surname
    Hsia         2456
    T'ien        2282
    Hs?          1611
    Kao          1577
    Maclean      1577
                 ... 
    Samaniego       1
    Lawley          1
    Bonwick         1
    Tennant         1
    Elkins          1
    Name: count, Length: 2797, dtype: int64 
    ---------------------------------
    
    Number of unique categories:  3
    Geography
    France     94215
    Spain      36213
    Germany    34606
    Name: count, dtype: int64 
    ---------------------------------
    
    Number of unique categories:  2
    Gender
    Male      93150
    Female    71884
    Name: count, dtype: int64 
    ---------------------------------
    
    


```python
plt.pie(df['Exited'].value_counts().values, explode=(0.1,0), labels=['Stayed','Churned'], autopct='%1.1f%%')
plt.title('Churn Distribution')
plt.show()
```


    
![png](/images/bank_churn_binary/output_15_0.png)
    


We see that this is a slightly imbalanced dataset with our target class making up ~21.2% of the samples.


```python
cat_cols
```




    Index(['Surname', 'Geography', 'Gender'], dtype='object')




```python
for col in cat_cols:
    if col == 'Surname':
        continue
    xtab = pd.crosstab(df[col], df['Exited'], normalize='index')
    xtab.plot(kind='bar', stacked=True, fontsize=10).legend(loc='lower right')
    plt.title(f'Percentage Distribution of Churn across {col}')
    plt.tight_layout()
    plt.show()
```


    
![png](/images/bank_churn_binary/output_18_0.png)
    



    
![png](/images/bank_churn_binary/output_18_1.png)
    


Here we see that a higher proportion of customers from Germany have churned, similar behaviour is seen in Female customers across the dataset. This suggests that the features will be useful in our predictive model.


```python
xtab = pd.crosstab(df['Tenure'], df['Exited'], normalize='index')
xtab.plot(kind='bar', stacked=True, fontsize=10).legend(loc='lower right')
plt.title(f'Percentage Distribution of Churn across Tenure')
plt.tight_layout()
plt.show()
```


    
![png](/images/bank_churn_binary/output_20_0.png)
    


New customer with 0 years with the bank has a slightly higher churn rate than the other customers.


```python
pd.DataFrame(df.groupby(['HasCrCard','IsActiveMember'])[['HasCrCard','IsActiveMember']].value_counts())
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
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">0.0</th>
      <th>0.0</th>
      <td>19646</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>20960</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1.0</th>
      <th>0.0</th>
      <td>63239</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>61189</td>
    </tr>
  </tbody>
</table>
</div>



There is an even distribution among customer that have a credit card and whether they are an active member (unable to verify what this means in the context of this dataset)

In this dataset ~50% of the customers who have a credit card is active, also 50% of the customers who do not have a credit card is active. This goes against the intuition that most customers with an active credit card with the bank will most likely be using it at least occassionally. I suspect this is due to the synthetic nature of the dataset.

In reality more clarity will be required for an ambiguous feature such as IsActiveMember, such as the definition and conditions surrounding a customer being considered active. With these information further feature engineering can possible be conducted to extract more useful information out of this feature.



```python
df.describe()
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
      <th>CustomerId</th>
      <th>CreditScore</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>165034.0000</td>
      <td>1.650340e+05</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
      <td>165034.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>82516.5000</td>
      <td>1.569201e+07</td>
      <td>656.454373</td>
      <td>38.125888</td>
      <td>5.020353</td>
      <td>55478.086689</td>
      <td>1.554455</td>
      <td>0.753954</td>
      <td>0.497770</td>
      <td>112574.822734</td>
      <td>0.211599</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47641.3565</td>
      <td>7.139782e+04</td>
      <td>80.103340</td>
      <td>8.867205</td>
      <td>2.806159</td>
      <td>62817.663278</td>
      <td>0.547154</td>
      <td>0.430707</td>
      <td>0.499997</td>
      <td>50292.865585</td>
      <td>0.408443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0000</td>
      <td>1.556570e+07</td>
      <td>350.000000</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.580000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>41258.2500</td>
      <td>1.563314e+07</td>
      <td>597.000000</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>74637.570000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>82516.5000</td>
      <td>1.569017e+07</td>
      <td>659.000000</td>
      <td>37.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>117948.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>123774.7500</td>
      <td>1.575682e+07</td>
      <td>710.000000</td>
      <td>42.000000</td>
      <td>7.000000</td>
      <td>119939.517500</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>155152.467500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>165033.0000</td>
      <td>1.581569e+07</td>
      <td>850.000000</td>
      <td>92.000000</td>
      <td>10.000000</td>
      <td>250898.090000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>199992.480000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
num_cols
```




    Index(['id', 'CustomerId', 'CreditScore', 'Age', 'Tenure', 'Balance',
           'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
           'Exited'],
          dtype='object')




```python
for col in num_cols:
    if col in ['id','Exited','HasCrCard','Tenure','NumOfProducts','IsActiveMember']:
        continue
    sns.violinplot(df, x='Exited', y=col)
    plt.title(f'{col} Distribution by Target')
    plt.show()
```


    
![png](/images/bank_churn_binary/output_26_0.png)
    



    
![png](/images/bank_churn_binary/output_26_1.png)
    



    
![png](/images/bank_churn_binary/output_26_2.png)
    



    
![png](/images/bank_churn_binary/output_26_3.png)
    



    
![png](/images/bank_churn_binary/output_26_4.png)
    


The numerical features all have similar distributions between their churned and non-churned customers except for Age where we see the distribution for churned customers is centred around an older age. (~45 years old vs ~35 years old for non-churned). 



```python
# Fitting labelencoders with all the labels from train and test dataset so the label mappings are consistent
test_labels = pd.read_csv('test.csv')[['Geography','Gender','Surname']]
set_geo = set(df['Geography']).union(test_labels['Geography'])
set_gender = set(df['Gender']).union(test_labels['Gender'])
set_surname = set(df['Surname']).union(test_labels['Surname'])
le_geo = LabelEncoder()
le_gender = LabelEncoder()
le_surname = LabelEncoder()
le_geo.fit(list(set_geo))
le_gender.fit(list(set_gender))
le_surname.fit(list(set_surname))

# Encoding categorical features
df['Geography'] = le_geo.transform(df['Geography'])
df['Gender'] = le_gender.transform(df['Gender'])
df['Surname'] = le_surname.transform(df['Surname'])
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
      <th>id</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>15674932</td>
      <td>1992</td>
      <td>668</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>181449.97</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>15749177</td>
      <td>1993</td>
      <td>627</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>49503.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>15694510</td>
      <td>1217</td>
      <td>678</td>
      <td>0</td>
      <td>1</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>184866.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15741417</td>
      <td>1341</td>
      <td>581</td>
      <td>0</td>
      <td>1</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>84560.88</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>15766172</td>
      <td>483</td>
      <td>716</td>
      <td>2</td>
      <td>1</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15068.83</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(df.corr(numeric_only=True))
plt.title('Correlation Heatmap of Features')
plt.show()
```


    
![png](/images/bank_churn_binary/output_29_0.png)
    


Looking at linear correlation coefficients we see that Age and NumOfProducts have the largest effect on churn

<a id='engineering'></a>
## 4. Feature Engineering


```python
# Feature engineering
df_eng = df.copy()

# Customer above age of retirement of the 3 countries in dataset
df_eng.loc[df_eng['Age']>=60, 'SeniorAge'] = 1
df_eng.loc[df_eng['Age']<60, 'SeniorAge'] = 0

# Customer is still young
df_eng.loc[df_eng['Age']<=35, 'Young'] = 1
df_eng.loc[df_eng['Age']>35, 'Young'] = 0

# Ratio of Balance and Estimated Salary
df_eng['Ratio_Bal_Sal'] = df_eng['Balance']/df_eng['EstimatedSalary']

# Ratio of Balance and Age
df_eng['Ratio_Bal_Age'] = df_eng['Balance']/df_eng['Age']

# Ratio of Estimated Salary and Age
df_eng['Ratio_Sal_Age'] = df_eng['EstimatedSalary']/df_eng['Age']

# Ratio of Tenure and NumOfProducts
df_eng['Ratio_Ten_Num'] = df_eng['Tenure']/df_eng['NumOfProducts']

# CreditScore bin
df_eng['Bin_CreditScore'] = pd.cut(df_eng['CreditScore'], bins=[0, 600, 700, 900], labels=[0,1,2]).astype(int)

# Age bin
df_eng['Bin_Age'] = df_eng['Age']//10

df_eng.head()
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>...</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
      <th>SeniorAge</th>
      <th>Young</th>
      <th>Ratio_Bal_Sal</th>
      <th>Ratio_Bal_Age</th>
      <th>Ratio_Sal_Age</th>
      <th>Ratio_Ten_Num</th>
      <th>Bin_CreditScore</th>
      <th>Bin_Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>15674932</td>
      <td>1992</td>
      <td>668</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>...</td>
      <td>181449.97</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5498.483939</td>
      <td>1.5</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>15749177</td>
      <td>1993</td>
      <td>627</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>...</td>
      <td>49503.50</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1500.106061</td>
      <td>0.5</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>15694510</td>
      <td>1217</td>
      <td>678</td>
      <td>0</td>
      <td>1</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>...</td>
      <td>184866.69</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4621.667250</td>
      <td>5.0</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15741417</td>
      <td>1341</td>
      <td>581</td>
      <td>0</td>
      <td>1</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>...</td>
      <td>84560.88</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.760655</td>
      <td>4378.898235</td>
      <td>2487.084706</td>
      <td>2.0</td>
      <td>0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>15766172</td>
      <td>483</td>
      <td>716</td>
      <td>2</td>
      <td>1</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>...</td>
      <td>15068.83</td>
      <td>0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>456.631212</td>
      <td>2.5</td>
      <td>2</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



Above are some simple features that were created to transform some of the features to help with our model's learning. These could be through inspiration and knowledge of the domain or simply random numerical transformations.


```python
df_eng.corr()[['Exited']].index
```




    Index(['id', 'CustomerId', 'Surname', 'CreditScore', 'Geography', 'Gender',
           'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
           'IsActiveMember', 'EstimatedSalary', 'Exited', 'SeniorAge', 'Young',
           'Ratio_Bal_Sal', 'Ratio_Bal_Age', 'Ratio_Sal_Age', 'Ratio_Ten_Num',
           'Bin_CreditScore', 'Bin_Age'],
          dtype='object')




```python
sns.heatmap(df_eng.corr()[['Exited']], annot=True, yticklabels=df_eng.corr()[['Exited']].index)
plt.title('Correlation Heatmap of Features')
plt.show()
```


    
![png](/images/bank_churn_binary/output_34_0.png)
    



```python
X_train = df_eng.drop(['id','CustomerId','Exited'], axis=1)
X_train.head()
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
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>SeniorAge</th>
      <th>Young</th>
      <th>Ratio_Bal_Sal</th>
      <th>Ratio_Bal_Age</th>
      <th>Ratio_Sal_Age</th>
      <th>Ratio_Ten_Num</th>
      <th>Bin_CreditScore</th>
      <th>Bin_Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1992</td>
      <td>668</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>181449.97</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5498.483939</td>
      <td>1.5</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1993</td>
      <td>627</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>49503.50</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1500.106061</td>
      <td>0.5</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1217</td>
      <td>678</td>
      <td>0</td>
      <td>1</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>184866.69</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4621.667250</td>
      <td>5.0</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1341</td>
      <td>581</td>
      <td>0</td>
      <td>1</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>84560.88</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.760655</td>
      <td>4378.898235</td>
      <td>2487.084706</td>
      <td>2.0</td>
      <td>0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>483</td>
      <td>716</td>
      <td>2</td>
      <td>1</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15068.83</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>456.631212</td>
      <td>2.5</td>
      <td>2</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train = df_eng['Exited']
y_train.head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: Exited, dtype: int64




```python
class AddClustersFeature(BaseEstimator, TransformerMixin):
    def __init__(self, clusters = 8): 
        self.clusters = clusters
        
           
    def fit(self, X, y=None):
        self.X=X
        self.model = KMeans(n_clusters = self.clusters)
        self.model.fit (self.X)
        return self
       
    def transform(self, X):
        self.X=X
        X_=X.copy() # avoiding modification of the original df
        X_ = pd.DataFrame(X_)
        X_['Clusters'] = self.model.predict(X_)
        X_.columns = X_.columns.astype(str)
        #print(X_.info())
        return X_
```


```python
# Sanity check that function to add cluster as a feature works
cluster_sanity = X_train.copy()
m = AddClustersFeature()
m.fit(cluster_sanity)
cluster_sanity = m.transform(cluster_sanity)
cluster_sanity.head()
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
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>SeniorAge</th>
      <th>Young</th>
      <th>Ratio_Bal_Sal</th>
      <th>Ratio_Bal_Age</th>
      <th>Ratio_Sal_Age</th>
      <th>Ratio_Ten_Num</th>
      <th>Bin_CreditScore</th>
      <th>Bin_Age</th>
      <th>Clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1992</td>
      <td>668</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>181449.97</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5498.483939</td>
      <td>1.5</td>
      <td>1</td>
      <td>3.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1993</td>
      <td>627</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>49503.50</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1500.106061</td>
      <td>0.5</td>
      <td>1</td>
      <td>3.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1217</td>
      <td>678</td>
      <td>0</td>
      <td>1</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>184866.69</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4621.667250</td>
      <td>5.0</td>
      <td>1</td>
      <td>4.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1341</td>
      <td>581</td>
      <td>0</td>
      <td>1</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>84560.88</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.760655</td>
      <td>4378.898235</td>
      <td>2487.084706</td>
      <td>2.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>483</td>
      <td>716</td>
      <td>2</td>
      <td>1</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15068.83</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>456.631212</td>
      <td>2.5</td>
      <td>2</td>
      <td>3.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



Through clustering of our data points, another engineered feature can be added to our dataset. Depending on the clustering algorithm it can provide different additional information about our data points. In this case we are using KMeans which attempts to group similar datapoints based on their Euclidean distances.

<a id='modelling'></a>
## 5. Modelling


```python
#################################################################
```


```python
# Instantiate models and params
logreg = LogisticRegression(random_state = 47)
logreg_params = {'C': np.logspace(-4, 4, 6),
                 'solver': ['lbfgs','newton-cg','sag','saga']
                }

rfc = RandomForestClassifier(random_state = 47)
rfc_params = {'n_estimators': [10,50,100],
              'min_samples_split': [2, 5, 10, 20]
             }

xgb = XGBClassifier(random_state = 47,
                    objective='binary:logistic',
                    metric='auc',
                    device = 'cuda', error_score='raise')
xgb_params = {'eta': [0.01,0.1,0.3],
              'max_depth': [3,6,9],
              'lambda': [0.3,0.6,1],
              'alpha': [0,0.1],
              'min_child_weight': [1,10,20],
              'colsample_bytree': [0.25,0.5,1]
             }
              
lgb = LGBMClassifier(random_state = 47,
                    objective='binary',
                    metric='auc',
                    verbosity=-1)
lgb_params = {'max_bin': [10,69,150,255,400],
              'max_depth': [3,6,9],
              'learning_rate': [ 0.01, 0.1],
              'lambda_l1': [0,0.1],
              'lambda_l2': [0,0.3,0.6],
              'num_leaves': [10,31,50]
             }

clfs = [
    ('Logistic Regression', logreg, logreg_params),
    ('Random Forest Classifier', rfc, rfc_params),
    ('XGBoost Classifier', xgb, xgb_params),
    ('LGBM Classifier', lgb, lgb_params)
]

scorers = {
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score),
    'roc_auc_score': make_scorer(roc_auc_score)
}
```


```python
# Compare model performance of (all features) vs (all w/o cluster features) vs (all w/o cluster+surname features) vs (w/o scaling) using xgb
X_train_nosurname = X_train.drop('Surname',axis=1)

pipeline = Pipeline(steps = [
    ('scaler', MinMaxScaler()),
    ('cluster', AddClustersFeature()),
    ('xgb', xgb)
])
pipeline_noclust = Pipeline(steps=[
    ('scaler', MinMaxScaler()),
    ('xgb', xgb)
])
pipeline_noscale = Pipeline(steps=[
    ('cluster', AddClustersFeature()),
    ('xgb', xgb)
])

pipelines = [('Pipeline', pipeline), ('Pipeline_No_Cluster', pipeline_noclust), ('Pipeline_No_Scale', pipeline_noscale)]
results = []
cv = KFold(n_splits=5)
for pipe_name, pipe in pipelines:
    scores = cross_val_score(estimator=pipe, X=X_train, y=y_train, cv=cv, scoring='roc_auc')
    results.append(scores)
    
    scores = cross_val_score(estimator=pipe, X=X_train_nosurname, y=y_train, cv=cv, scoring='roc_auc')
    results.append(scores)
```

Here we are fitting an XGBoost model in several different scenarios to find our best performing scenario. Below are the scenarios:
1. Dataset is scaled and cluster feature added
2. Dataset is scaled
3. Dataset has cluster feature added
This is repeated for training data with and without Surname feature for a total of 6 different scenarios.


```python
best_result = [-1, 0]
for idx, result in enumerate(results):
    mean_result = np.mean(result)
    print(f'Pipeline: {idx}, Mean ROC AUC: {mean_result}')
    if mean_result > best_result[1]:
        best_result = [idx, mean_result]
print(f' Best pipeline is {best_result[0]} with a mean score of {best_result[1]}')
# Best performer is dataset including Surname, with feature scaling and additional cluster feature included
```

    Pipeline: 0, Mean ROC AUC: 0.8888409253562118
    Pipeline: 1, Mean ROC AUC: 0.8859394416907461
    Pipeline: 2, Mean ROC AUC: 0.8886329403159834
    Pipeline: 3, Mean ROC AUC: 0.8858966559943475
    Pipeline: 4, Mean ROC AUC: 0.8887585013508655
    Pipeline: 5, Mean ROC AUC: 0.8862002696202766
     Best pipeline is 0 with a mean score of 0.8888409253562118
    

Best performer after 5 fold cross validation is Dataset+Surname feature with feature scaling and cluster feature included. This is the scenario that we will be tuning our final models on.


```python
# Pipeline
results = []
for clf_name, clf, clf_params in clfs:
    gs = GridSearchCV(estimator=clf, 
                      param_grid=clf_params,
                      scoring=scorers,
                      refit='roc_auc_score',
                      verbose=2,
                      error_score='raise'
                     )
    pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('cluster', AddClustersFeature()),
        ('classifier', gs),
    ])
    pipeline.fit(X_train, y_train)
    result = [clf_name, gs.best_params_, gs.best_score_, gs.cv_results_['mean_test_f1_score'][gs.best_index_], gs.cv_results_['mean_test_accuracy_score'][gs.best_index_]]
    results.append(result)
result_df = pd.DataFrame(results, columns=['Name','Parameters','ROCAUC','F1','Accuracy'])
result_df.head()
```

    Fitting 5 folds for each of 24 candidates, totalling 120 fits
    Fitting 5 folds for each of 12 candidates, totalling 60 fits
    Fitting 5 folds for each of 486 candidates, totalling 2430 fits
    Fitting 5 folds for each of 540 candidates, totalling 2700 fits
    
    




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
      <th>Name</th>
      <th>Parameters</th>
      <th>ROCAUC</th>
      <th>F1</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>{'C': 6.309573444801943, 'solver': 'lbfgs'}</td>
      <td>0.671971</td>
      <td>0.498950</td>
      <td>0.833295</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest Classifier</td>
      <td>{'min_samples_split': 20, 'n_estimators': 100}</td>
      <td>0.742522</td>
      <td>0.622877</td>
      <td>0.863489</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBoost Classifier</td>
      <td>{'alpha': 0, 'colsample_bytree': 1, 'eta': 0.3...</td>
      <td>0.755960</td>
      <td>0.641102</td>
      <td>0.866179</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LGBM Classifier</td>
      <td>{'lambda_l1': 0, 'lambda_l2': 0.6, 'learning_r...</td>
      <td>0.755667</td>
      <td>0.641445</td>
      <td>0.866840</td>
    </tr>
  </tbody>
</table>
</div>



Out of the 4 models tuned, XGB and LGBM classifiers have the best performance. As their performance are similar, we shall go one step further and create a final voting classifier to make use of both best performers.


```python
result_df[result_df['Name'].isin(['XGBoost Classifier','LGBM Classifier'])]['Parameters'].tolist()
```




    [{'alpha': 0,
      'colsample_bytree': 1,
      'eta': 0.3,
      'lambda': 0.6,
      'max_depth': 6,
      'min_child_weight': 1},
     {'lambda_l1': 0,
      'lambda_l2': 0.6,
      'learning_rate': 0.1,
      'max_bin': 400,
      'max_depth': 6,
      'num_leaves': 31}]




```python
########################################################################################
```


```python
xgb = XGBClassifier(random_state = 47,
                    objective='binary:logistic',
                    metric='auc',
                    device = 'cuda',
                    alpha= 0,
                    colsample_bytree= 1,
                    eta= 0.3,
                    reg_lambda= 0.6,
                    max_depth= 6,
                    min_child_weight= 1,
                    error_score='raise')
              
lgb = LGBMClassifier(random_state = 47,
                    objective='binary',
                    metric='auc',
                    verbosity=-1,
                    lambda_l1= 0,
                    lambda_l2= 0.6,
                    learning_rate= 0.1,
                    max_bin= 400,
                    max_depth= 6,
                    num_leaves= 31)

vc = VotingClassifier(estimators=[('xgb',xgb),('lgb',lgb)], voting='soft')
```


```python
X_train.head()
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
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>SeniorAge</th>
      <th>Young</th>
      <th>Ratio_Bal_Sal</th>
      <th>Ratio_Bal_Age</th>
      <th>Ratio_Sal_Age</th>
      <th>Ratio_Ten_Num</th>
      <th>Bin_CreditScore</th>
      <th>Bin_Age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1992</td>
      <td>668</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>181449.97</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5498.483939</td>
      <td>1.5</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1993</td>
      <td>627</td>
      <td>0</td>
      <td>1</td>
      <td>33.0</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>49503.50</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1500.106061</td>
      <td>0.5</td>
      <td>1</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1217</td>
      <td>678</td>
      <td>0</td>
      <td>1</td>
      <td>40.0</td>
      <td>10</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>184866.69</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4621.667250</td>
      <td>5.0</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1341</td>
      <td>581</td>
      <td>0</td>
      <td>1</td>
      <td>34.0</td>
      <td>2</td>
      <td>148882.54</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>84560.88</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.760655</td>
      <td>4378.898235</td>
      <td>2487.084706</td>
      <td>2.0</td>
      <td>0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>483</td>
      <td>716</td>
      <td>2</td>
      <td>1</td>
      <td>33.0</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>15068.83</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>456.631212</td>
      <td>2.5</td>
      <td>2</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Scaling and adding cluster on full training data
scaler = MinMaxScaler()
X_train_final = X_train.copy()
X_train_final = scaler.fit_transform(X_train_final)
X_train_final = AddClustersFeature().fit_transform(X_train_final)
X_train_final
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>Clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.689751</td>
      <td>0.636</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.202703</td>
      <td>0.3</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.907279</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.496366</td>
      <td>0.15</td>
      <td>0.5</td>
      <td>0.250</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.690097</td>
      <td>0.554</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.202703</td>
      <td>0.1</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.247483</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.135405</td>
      <td>0.05</td>
      <td>0.5</td>
      <td>0.250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.421399</td>
      <td>0.656</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.297297</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.924364</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.417210</td>
      <td>0.50</td>
      <td>0.5</td>
      <td>0.375</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.464335</td>
      <td>0.462</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.216216</td>
      <td>0.2</td>
      <td>0.593398</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.422787</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000137</td>
      <td>0.424012</td>
      <td>0.224506</td>
      <td>0.20</td>
      <td>0.0</td>
      <td>0.250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.167244</td>
      <td>0.732</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.202703</td>
      <td>0.5</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.075293</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041203</td>
      <td>0.25</td>
      <td>1.0</td>
      <td>0.250</td>
      <td>3</td>
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
    </tr>
    <tr>
      <th>165029</th>
      <td>0.608726</td>
      <td>0.634</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.202703</td>
      <td>0.2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.659179</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.360636</td>
      <td>0.20</td>
      <td>0.5</td>
      <td>0.250</td>
      <td>1</td>
    </tr>
    <tr>
      <th>165030</th>
      <td>0.687673</td>
      <td>0.884</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.229730</td>
      <td>0.3</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.659177</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.340026</td>
      <td>0.30</td>
      <td>1.0</td>
      <td>0.250</td>
      <td>6</td>
    </tr>
    <tr>
      <th>165031</th>
      <td>0.419321</td>
      <td>0.430</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.175676</td>
      <td>0.5</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.637151</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.371075</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.250</td>
      <td>3</td>
    </tr>
    <tr>
      <th>165032</th>
      <td>0.420706</td>
      <td>0.408</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.162162</td>
      <td>0.7</td>
      <td>0.643819</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.355841</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000176</td>
      <td>0.521378</td>
      <td>0.214156</td>
      <td>0.70</td>
      <td>0.0</td>
      <td>0.250</td>
      <td>1</td>
    </tr>
    <tr>
      <th>165033</th>
      <td>0.916898</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.175676</td>
      <td>0.1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.307880</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.179316</td>
      <td>0.10</td>
      <td>1.0</td>
      <td>0.250</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
<p>165034 rows × 20 columns</p>
</div>




```python
cross_val_score(vc, X_train_final, y_train, scoring='roc_auc')
```




    array([0.8940815 , 0.88922709, 0.89204022, 0.89081926, 0.8903584 ])



Above shows our expected model performance if our model did not overfit and trainin/test data share similar distributions


```python
vc.fit(X_train_final, y_train)
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>VotingClassifier(estimators=[(&#x27;xgb&#x27;,
                              XGBClassifier(alpha=0, base_score=None,
                                            booster=None, callbacks=None,
                                            colsample_bylevel=None,
                                            colsample_bynode=None,
                                            colsample_bytree=1, device=&#x27;cuda&#x27;,
                                            early_stopping_rounds=None,
                                            enable_categorical=False,
                                            error_score=&#x27;raise&#x27;, eta=0.3,
                                            eval_metric=None,
                                            feature_types=None, gamma=None,
                                            grow_policy=None,
                                            importance_type=None,
                                            inter...
                                            learning_rate=None, max_bin=None,
                                            max_cat_threshold=None,
                                            max_cat_to_onehot=None,
                                            max_delta_step=None, max_depth=6,
                                            max_leaves=None, metric=&#x27;auc&#x27;,
                                            min_child_weight=1, missing=nan,
                                            monotone_constraints=None,
                                            multi_strategy=None, ...)),
                             (&#x27;lgb&#x27;,
                              LGBMClassifier(lambda_l1=0, lambda_l2=0.6,
                                             max_bin=400, max_depth=6,
                                             metric=&#x27;auc&#x27;, objective=&#x27;binary&#x27;,
                                             random_state=47, verbosity=-1))],
                 voting=&#x27;soft&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">VotingClassifier</label><div class="sk-toggleable__content"><pre>VotingClassifier(estimators=[(&#x27;xgb&#x27;,
                              XGBClassifier(alpha=0, base_score=None,
                                            booster=None, callbacks=None,
                                            colsample_bylevel=None,
                                            colsample_bynode=None,
                                            colsample_bytree=1, device=&#x27;cuda&#x27;,
                                            early_stopping_rounds=None,
                                            enable_categorical=False,
                                            error_score=&#x27;raise&#x27;, eta=0.3,
                                            eval_metric=None,
                                            feature_types=None, gamma=None,
                                            grow_policy=None,
                                            importance_type=None,
                                            inter...
                                            learning_rate=None, max_bin=None,
                                            max_cat_threshold=None,
                                            max_cat_to_onehot=None,
                                            max_delta_step=None, max_depth=6,
                                            max_leaves=None, metric=&#x27;auc&#x27;,
                                            min_child_weight=1, missing=nan,
                                            monotone_constraints=None,
                                            multi_strategy=None, ...)),
                             (&#x27;lgb&#x27;,
                              LGBMClassifier(lambda_l1=0, lambda_l2=0.6,
                                             max_bin=400, max_depth=6,
                                             metric=&#x27;auc&#x27;, objective=&#x27;binary&#x27;,
                                             random_state=47, verbosity=-1))],
                 voting=&#x27;soft&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>xgb</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(alpha=0, base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None, colsample_bytree=1,
              device=&#x27;cuda&#x27;, early_stopping_rounds=None,
              enable_categorical=False, error_score=&#x27;raise&#x27;, eta=0.3,
              eval_metric=None, feature_types=None, gamma=None,
              grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=6, max_leaves=None, metric=&#x27;auc&#x27;,
              min_child_weight=1, missing=nan, monotone_constraints=None,
              multi_strategy=None, ...)</pre></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><label>lgb</label></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LGBMClassifier</label><div class="sk-toggleable__content"><pre>LGBMClassifier(lambda_l1=0, lambda_l2=0.6, max_bin=400, max_depth=6,
               metric=&#x27;auc&#x27;, objective=&#x27;binary&#x27;, random_state=47, verbosity=-1)</pre></div></div></div></div></div></div></div></div></div></div>




```python
# Preparing test data for prediction
test_df = pd.read_csv('test.csv')
test_df.head()
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
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165034</td>
      <td>15773898</td>
      <td>Lucchese</td>
      <td>586</td>
      <td>France</td>
      <td>Female</td>
      <td>23.0</td>
      <td>2</td>
      <td>0.00</td>
      <td>2</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>160976.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>165035</td>
      <td>15782418</td>
      <td>Nott</td>
      <td>683</td>
      <td>France</td>
      <td>Female</td>
      <td>46.0</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>72549.27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>165036</td>
      <td>15807120</td>
      <td>K?</td>
      <td>656</td>
      <td>France</td>
      <td>Female</td>
      <td>34.0</td>
      <td>7</td>
      <td>0.00</td>
      <td>2</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>138882.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>165037</td>
      <td>15808905</td>
      <td>O'Donnell</td>
      <td>681</td>
      <td>France</td>
      <td>Male</td>
      <td>36.0</td>
      <td>8</td>
      <td>0.00</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113931.57</td>
    </tr>
    <tr>
      <th>4</th>
      <td>165038</td>
      <td>15607314</td>
      <td>Higgins</td>
      <td>752</td>
      <td>Germany</td>
      <td>Male</td>
      <td>38.0</td>
      <td>10</td>
      <td>121263.62</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>139431.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.shape
```




    (110023, 13)




```python
test_df['Geography'].unique()
```




    array(['France', 'Germany', 'Spain'], dtype=object)




```python
X_train_final.shape
```




    (165034, 20)




```python
# Encoding categorical features
test_df['Geography'] = le_geo.transform(test_df['Geography'])
test_df['Gender'] = le_gender.transform(test_df['Gender'])
test_df['Surname'] = le_surname.transform(test_df['Surname'])
test_df.head()

# Customer above age of retirement of the 3 countries in dataset
test_df.loc[test_df['Age']>=60, 'SeniorAge'] = 1
test_df.loc[test_df['Age']<60, 'SeniorAge'] = 0

# Customer is still young
test_df.loc[test_df['Age']<=35, 'Young'] = 1
test_df.loc[test_df['Age']>35, 'Young'] = 0

# Ratio of Balance and Estimated Salary
test_df['Ratio_Bal_Sal'] = test_df['Balance']/test_df['EstimatedSalary']

# Ratio of Balance and Age
test_df['Ratio_Bal_Age'] = test_df['Balance']/test_df['Age']

# Ratio of Estimated Salary and Age
test_df['Ratio_Sal_Age'] = test_df['EstimatedSalary']/test_df['Age']

# Ratio of Tenure and NumOfProducts
test_df['Ratio_Ten_Num'] = test_df['Tenure']/test_df['NumOfProducts']

# CreditScore bin
test_df['Bin_CreditScore'] = pd.cut(test_df['CreditScore'], bins=[0, 600, 700, 900], labels=[0,1,2]).astype(int)

# Age bin
test_df['Bin_Age'] = test_df['Age']//10

# Save id to join with final predictions
idx =  test_df['id']

X_test = test_df.drop(['id','CustomerId'], axis=1)

# Scale features and add clusters
X_test = scaler.transform(X_test)
X_test = m.transform(X_test)
X_test.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>Clusters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.547438</td>
      <td>0.472</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.067568</td>
      <td>0.2</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.804903</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.631827</td>
      <td>0.10</td>
      <td>0.0</td>
      <td>0.125</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.670014</td>
      <td>0.666</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.378378</td>
      <td>0.2</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.362723</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142361</td>
      <td>0.20</td>
      <td>0.5</td>
      <td>0.375</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.460873</td>
      <td>0.612</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.216216</td>
      <td>0.7</td>
      <td>0.000000</td>
      <td>0.333333</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.694419</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.368740</td>
      <td>0.35</td>
      <td>0.5</td>
      <td>0.250</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.676939</td>
      <td>0.662</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.243243</td>
      <td>0.8</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.569654</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.285685</td>
      <td>0.80</td>
      <td>0.5</td>
      <td>0.250</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.399238</td>
      <td>0.804</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.270270</td>
      <td>1.0</td>
      <td>0.483318</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.697164</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000068</td>
      <td>0.309001</td>
      <td>0.331227</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>0.250</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Make and export predictions in format for submission
y_pred = vc.predict_proba(X_test)[:, 1]
predictions = pd.concat([idx, pd.Series(y_pred)], axis=1)
predictions.columns = 'id','Exited'
predictions.to_csv('predictions_.csv', index=False)
predictions
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
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>165034</td>
      <td>0.043242</td>
    </tr>
    <tr>
      <th>1</th>
      <td>165035</td>
      <td>0.856776</td>
    </tr>
    <tr>
      <th>2</th>
      <td>165036</td>
      <td>0.027904</td>
    </tr>
    <tr>
      <th>3</th>
      <td>165037</td>
      <td>0.242382</td>
    </tr>
    <tr>
      <th>4</th>
      <td>165038</td>
      <td>0.358144</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>110018</th>
      <td>275052</td>
      <td>0.030970</td>
    </tr>
    <tr>
      <th>110019</th>
      <td>275053</td>
      <td>0.154990</td>
    </tr>
    <tr>
      <th>110020</th>
      <td>275054</td>
      <td>0.023756</td>
    </tr>
    <tr>
      <th>110021</th>
      <td>275055</td>
      <td>0.160589</td>
    </tr>
    <tr>
      <th>110022</th>
      <td>275056</td>
      <td>0.196556</td>
    </tr>
  </tbody>
</table>
<p>110023 rows × 2 columns</p>
</div>



Our predictions currently has a public score of 0.88895 after submission, which is close to what we have expected.

<a id='conclusion'></a>
## 6. Conclusion
In this notebook we have explored a synthetic bank churn dataset and briefly discussed what could be done differently in a real world scenario. With a churn rate of ~21% our model has an ROCAUC of 0.88895 which appears to be a relatively good performance that beats out a random guess strategy. With some domain knowledge or time investment, one can create certain conditional strategies that predicts customer churn. Further analysis between the predictive model and those conditional strategies will then tell us if the model performance is indeed worthwhile.
