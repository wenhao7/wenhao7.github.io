---
layout: post
title:  "Multi-Class Prediction of Obesity Risk"
date:   2024-03-13 22:31:09 +0800
category: [data_analysis, visualization, machine_learning]
tag: [numpy, pandas, seaborn, matplotlib, scikit-learn, classification]
summary: "In this notebook we take a look at a [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s4e2) competition where users submit their predictions for a multi-class classification problem on the sample's weight class."
---

***
## Contents
1. [Introduction](#intro)
2. [Data Cleaning](#cleaning)
3. [Data Exploration](#eda)
4. [Feature Engineering](#engineering)
5. [Modelling](#modelling)
6. [Conclusion](#conclusion)
   
***

## 1. Introduction <a id='intro'></a>
In this notebook we take a look at a [Kaggle Playground Series](https://www.kaggle.com/competitions/playground-series-s4e2) competition where users submit their predictions for a multi-class classification problem on the sample's weight class.


<a id='cleaning'></a>
## 2. Data Cleaning


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, auc, roc_curve, roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')
```


```python
# Data import and cleaning
```


```python
data = pd.read_csv('train.csv')
data.head()
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
      <th>Gender</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>family_history_with_overweight</th>
      <th>FAVC</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>SMOKE</th>
      <th>CH2O</th>
      <th>SCC</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Male</td>
      <td>24.443011</td>
      <td>1.699998</td>
      <td>81.669950</td>
      <td>yes</td>
      <td>yes</td>
      <td>2.000000</td>
      <td>2.983297</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.763573</td>
      <td>no</td>
      <td>0.000000</td>
      <td>0.976473</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Overweight_Level_II</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Female</td>
      <td>18.000000</td>
      <td>1.560000</td>
      <td>57.000000</td>
      <td>yes</td>
      <td>yes</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>Frequently</td>
      <td>no</td>
      <td>2.000000</td>
      <td>no</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>no</td>
      <td>Automobile</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Female</td>
      <td>18.000000</td>
      <td>1.711460</td>
      <td>50.165754</td>
      <td>yes</td>
      <td>yes</td>
      <td>1.880534</td>
      <td>1.411685</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>1.910378</td>
      <td>no</td>
      <td>0.866045</td>
      <td>1.673584</td>
      <td>no</td>
      <td>Public_Transportation</td>
      <td>Insufficient_Weight</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Female</td>
      <td>20.952737</td>
      <td>1.710730</td>
      <td>131.274851</td>
      <td>yes</td>
      <td>yes</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>1.674061</td>
      <td>no</td>
      <td>1.467863</td>
      <td>0.780199</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Obesity_Type_III</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Male</td>
      <td>31.641081</td>
      <td>1.914186</td>
      <td>93.798055</td>
      <td>yes</td>
      <td>yes</td>
      <td>2.679664</td>
      <td>1.971472</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>1.979848</td>
      <td>no</td>
      <td>1.967973</td>
      <td>0.931721</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Overweight_Level_II</td>
    </tr>
  </tbody>
</table>
</div>



Descriptions of the columns within the dataset:
1. Frequent consumption of high caloric food (FAVC)
2. Frequency of consumption of vegetables (FCVC)
3. Number of main meals (NCP)
4. Consumption of food between meals (CAEC)
5. Consumption of water daily (CH20)
6. and Consumption of alcohol (CALC)
7. The attributes related with the physical condition are: Calories consumption monitoring (SCC)
8. Physical activity frequency (FAF)
9. Time using technology devices (TUE)
10. Transportation used (MTRANS)


```python
data.info()

```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20758 entries, 0 to 20757
    Data columns (total 18 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   id                              20758 non-null  int64  
     1   Gender                          20758 non-null  object 
     2   Age                             20758 non-null  float64
     3   Height                          20758 non-null  float64
     4   Weight                          20758 non-null  float64
     5   family_history_with_overweight  20758 non-null  object 
     6   FAVC                            20758 non-null  object 
     7   FCVC                            20758 non-null  float64
     8   NCP                             20758 non-null  float64
     9   CAEC                            20758 non-null  object 
     10  SMOKE                           20758 non-null  object 
     11  CH2O                            20758 non-null  float64
     12  SCC                             20758 non-null  object 
     13  FAF                             20758 non-null  float64
     14  TUE                             20758 non-null  float64
     15  CALC                            20758 non-null  object 
     16  MTRANS                          20758 non-null  object 
     17  NObeyesdad                      20758 non-null  object 
    dtypes: float64(8), int64(1), object(9)
    memory usage: 2.9+ MB
    




    '\nFrequent consumption of high caloric food (FAVC)\nFrequency of consumption of vegetables (FCVC)\nNumber of main meals (NCP)\nConsumption of food between meals (CAEC)\nConsumption of water daily (CH20)\nand Consumption of alcohol (CALC)\nThe attributes related with the physical condition are: Calories consumption monitoring (SCC)\nPhysical activity frequency (FAF)\nTime using technology devices (TUE)\nTransportation used (MTRANS)\n'




```python
data = data.drop('id',axis=1)
data.head()
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
      <th>Gender</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>family_history_with_overweight</th>
      <th>FAVC</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>SMOKE</th>
      <th>CH2O</th>
      <th>SCC</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>MTRANS</th>
      <th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Male</td>
      <td>24.443011</td>
      <td>1.699998</td>
      <td>81.669950</td>
      <td>yes</td>
      <td>yes</td>
      <td>2.000000</td>
      <td>2.983297</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>2.763573</td>
      <td>no</td>
      <td>0.000000</td>
      <td>0.976473</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Overweight_Level_II</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Female</td>
      <td>18.000000</td>
      <td>1.560000</td>
      <td>57.000000</td>
      <td>yes</td>
      <td>yes</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>Frequently</td>
      <td>no</td>
      <td>2.000000</td>
      <td>no</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>no</td>
      <td>Automobile</td>
      <td>Normal_Weight</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Female</td>
      <td>18.000000</td>
      <td>1.711460</td>
      <td>50.165754</td>
      <td>yes</td>
      <td>yes</td>
      <td>1.880534</td>
      <td>1.411685</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>1.910378</td>
      <td>no</td>
      <td>0.866045</td>
      <td>1.673584</td>
      <td>no</td>
      <td>Public_Transportation</td>
      <td>Insufficient_Weight</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Female</td>
      <td>20.952737</td>
      <td>1.710730</td>
      <td>131.274851</td>
      <td>yes</td>
      <td>yes</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>1.674061</td>
      <td>no</td>
      <td>1.467863</td>
      <td>0.780199</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Obesity_Type_III</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Male</td>
      <td>31.641081</td>
      <td>1.914186</td>
      <td>93.798055</td>
      <td>yes</td>
      <td>yes</td>
      <td>2.679664</td>
      <td>1.971472</td>
      <td>Sometimes</td>
      <td>no</td>
      <td>1.979848</td>
      <td>no</td>
      <td>1.967973</td>
      <td>0.931721</td>
      <td>Sometimes</td>
      <td>Public_Transportation</td>
      <td>Overweight_Level_II</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (20758, 17)




```python
data.duplicated().sum()
```




    0




```python
data.isna().sum()
```




    Gender                            0
    Age                               0
    Height                            0
    Weight                            0
    family_history_with_overweight    0
    FAVC                              0
    FCVC                              0
    NCP                               0
    CAEC                              0
    SMOKE                             0
    CH2O                              0
    SCC                               0
    FAF                               0
    TUE                               0
    CALC                              0
    MTRANS                            0
    NObeyesdad                        0
    dtype: int64



## 3. Exploration and Visualization
<a id='eda'></a>


```python
data.describe()
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
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>23.841804</td>
      <td>1.700245</td>
      <td>87.887768</td>
      <td>2.445908</td>
      <td>2.761332</td>
      <td>2.029418</td>
      <td>0.981747</td>
      <td>0.616756</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.688072</td>
      <td>0.087312</td>
      <td>26.379443</td>
      <td>0.533218</td>
      <td>0.705375</td>
      <td>0.608467</td>
      <td>0.838302</td>
      <td>0.602113</td>
    </tr>
    <tr>
      <th>min</th>
      <td>14.000000</td>
      <td>1.450000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>20.000000</td>
      <td>1.631856</td>
      <td>66.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>1.792022</td>
      <td>0.008013</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>22.815416</td>
      <td>1.700000</td>
      <td>84.064875</td>
      <td>2.393837</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.573887</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>26.000000</td>
      <td>1.762887</td>
      <td>111.600553</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.549617</td>
      <td>1.587406</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>61.000000</td>
      <td>1.975663</td>
      <td>165.057269</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
num_cols = data.select_dtypes(exclude=['object']).columns
```


```python
for col in data.select_dtypes('object').columns:
    print(data[col].value_counts())
    print("---------------\n")
```

    Gender
    Female    10422
    Male      10336
    Name: count, dtype: int64
    ---------------
    
    family_history_with_overweight
    yes    17014
    no      3744
    Name: count, dtype: int64
    ---------------
    
    FAVC
    yes    18982
    no      1776
    Name: count, dtype: int64
    ---------------
    
    CAEC
    Sometimes     17529
    Frequently     2472
    Always          478
    no              279
    Name: count, dtype: int64
    ---------------
    
    SMOKE
    no     20513
    yes      245
    Name: count, dtype: int64
    ---------------
    
    SCC
    no     20071
    yes      687
    Name: count, dtype: int64
    ---------------
    
    CALC
    Sometimes     15066
    no             5163
    Frequently      529
    Name: count, dtype: int64
    ---------------
    
    MTRANS
    Public_Transportation    16687
    Automobile                3534
    Walking                    467
    Motorbike                   38
    Bike                        32
    Name: count, dtype: int64
    ---------------
    
    NObeyesdad
    Obesity_Type_III       4046
    Obesity_Type_II        3248
    Normal_Weight          3082
    Obesity_Type_I         2910
    Insufficient_Weight    2523
    Overweight_Level_II    2522
    Overweight_Level_I     2427
    Name: count, dtype: int64
    ---------------
    
    

We see that there are some ordinal data
1. Consumption of food between meals (CAEC)
2. Consumption of alcohol (CALC)

and some nominal data
1. Transportation used (MTRANS)

later in the notebook we will encode these categorical features using different techniques to account for the ordinality.


```python
"""
•Underweight Less than 18.5
•Normal 18.5 to 24.9
•Overweight 25.0 to 29.9
•Obesity I 30.0 to 34.9
•Obesity II 35.0 to 39.9
•Obesity III Higher than 40
"""
```




    '\n•Underweight Less than 18.5\n•Normal 18.5 to 24.9\n•Overweight 25.0 to 29.9\n•Obesity I 30.0 to 34.9\n•Obesity II 35.0 to 39.9\n•Obesity III Higher than 40\n'




```python
sns.countplot(data=data, x='NObeyesdad', hue='Gender', order=['Insufficient_Weight','Normal_Weight','Overweight_Level_I','Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'])
plt.xticks(rotation=45, horizontalalignment='right')
plt.title("Target class by Gender")
plt.tight_layout()
plt.show()
```


    
![png](/images/multiclass_obesity/output_16_0.png)
    


Looking at the distribution of weightclass by gender we see that Obesity Type II is almost entirely made up of Male samples and Obesity Type III is almost entirely made up of Female samples.

This can be attributed to the synthetic nature of the dataset, and assuming that the test data will be generated with the same distribution this should not impact the performance of our model.


```python
sns.histplot(data=data, x='Age', hue='Gender', log_scale=(False,True))
plt.title("Distribution of samples by Age and Gender")
plt.show() # Most samples are between 20~40 years old
```

    C:\Users\wenhao\anaconda3\envs\ML\Lib\site-packages\seaborn\_oldcore.py:1119: FutureWarning: use_inf_as_na option is deprecated and will be removed in a future version. Convert inf values to NaN before operating instead.
      with pd.option_context('mode.use_inf_as_na', True):
    C:\Users\wenhao\anaconda3\envs\ML\Lib\site-packages\seaborn\_oldcore.py:1075: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    C:\Users\wenhao\anaconda3\envs\ML\Lib\site-packages\seaborn\_oldcore.py:1075: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    C:\Users\wenhao\anaconda3\envs\ML\Lib\site-packages\seaborn\_oldcore.py:1075: FutureWarning: When grouping with a length-1 list-like, you will need to pass a length-1 tuple to get_group in a future version of pandas. Pass `(name,)` instead of `name` to silence this warning.
      data_subset = grouped_data.get_group(pd_key)
    


    
![png](/images/multiclass_obesity/output_18_1.png)
    


We see most samples fall between 20 ~ 40 years old, take note that this y axis is in log scale making it look more uniform than it really is.


```python
sns.countplot(data=data, x='NObeyesdad', hue='FAVC', order=['Insufficient_Weight','Normal_Weight','Overweight_Level_I','Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'])
plt.xticks(rotation=45, horizontalalignment='right')
plt.title("Target class by Frequent Consumption of High Calorie Food")
plt.tight_layout()
plt.show() # Different CAEC distribution for target classes, almost all obese samples falls under "Sometimes" category.
```


    
![png](/images/multiclass_obesity/output_20_0.png)
    



```python
sns.countplot(data=data, x='NObeyesdad', hue='CAEC', order=['Insufficient_Weight','Normal_Weight','Overweight_Level_I','Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'])
plt.xticks(rotation=45, horizontalalignment='right')
plt.title("Target class by Consumption of food between meals")
plt.tight_layout()
plt.show() # Almost no obese samples fall under Walking/Motorbike/Bike
```


    
![png](/images/multiclass_obesity/output_21_0.png)
    



```python
sns.countplot(data=data, x='NObeyesdad', hue='MTRANS', order=['Insufficient_Weight','Normal_Weight','Overweight_Level_I','Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'])
plt.xticks(rotation=45, horizontalalignment='right')
plt.title("Target class by Transportation Used")
plt.tight_layout()
plt.show() # Almost no obese sample monitor their calories consumption
```


    
![png](/images/multiclass_obesity/output_22_0.png)
    



```python
sns.countplot(data=data, x='NObeyesdad', hue='SCC', order=['Insufficient_Weight','Normal_Weight','Overweight_Level_I','Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III'])
plt.xticks(rotation=45, horizontalalignment='right')
plt.title("Target class by whether respondent monitors Calories Consumption")
plt.tight_layout()
plt.show()
```


    
![png](/images/multiclass_obesity/output_23_0.png)
    



```python
# Feature encoding
"""
Ordinal encode
Consumption of food between meals (CAEC)
Consumption of alcohol (CALC)
Target columns (NObeyesdad)

Nominal encode
Transportation used (MTRANS)
"""
```




    '\nOrdinal encode\nConsumption of food between meals (CAEC)\nConsumption of alcohol (CALC)\nTarget columns (NObeyesdad)\n\nNominal encode\nTransportation used (MTRANS)\n'




```python
data.select_dtypes('object').columns
```




    Index(['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
           'SCC', 'CALC', 'MTRANS', 'NObeyesdad'],
          dtype='object')




```python
# Dummy encoding
dummy_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE',
       'SCC', 'MTRANS']
dummy_var = pd.get_dummies(data[dummy_cols], drop_first=True, dtype=int)
dummy_var.head()
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
      <th>Gender_Male</th>
      <th>family_history_with_overweight_yes</th>
      <th>FAVC_yes</th>
      <th>SMOKE_yes</th>
      <th>SCC_yes</th>
      <th>MTRANS_Bike</th>
      <th>MTRANS_Motorbike</th>
      <th>MTRANS_Public_Transportation</th>
      <th>MTRANS_Walking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = pd.concat([data, dummy_var], axis=1)
data = data.drop(dummy_cols,axis=1)
data.head()
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
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>NObeyesdad</th>
      <th>Gender_Male</th>
      <th>family_history_with_overweight_yes</th>
      <th>FAVC_yes</th>
      <th>SMOKE_yes</th>
      <th>SCC_yes</th>
      <th>MTRANS_Bike</th>
      <th>MTRANS_Motorbike</th>
      <th>MTRANS_Public_Transportation</th>
      <th>MTRANS_Walking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.443011</td>
      <td>1.699998</td>
      <td>81.669950</td>
      <td>2.000000</td>
      <td>2.983297</td>
      <td>Sometimes</td>
      <td>2.763573</td>
      <td>0.000000</td>
      <td>0.976473</td>
      <td>Sometimes</td>
      <td>Overweight_Level_II</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.000000</td>
      <td>1.560000</td>
      <td>57.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>Frequently</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>no</td>
      <td>Normal_Weight</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.000000</td>
      <td>1.711460</td>
      <td>50.165754</td>
      <td>1.880534</td>
      <td>1.411685</td>
      <td>Sometimes</td>
      <td>1.910378</td>
      <td>0.866045</td>
      <td>1.673584</td>
      <td>no</td>
      <td>Insufficient_Weight</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.952737</td>
      <td>1.710730</td>
      <td>131.274851</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>Sometimes</td>
      <td>1.674061</td>
      <td>1.467863</td>
      <td>0.780199</td>
      <td>Sometimes</td>
      <td>Obesity_Type_III</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31.641081</td>
      <td>1.914186</td>
      <td>93.798055</td>
      <td>2.679664</td>
      <td>1.971472</td>
      <td>Sometimes</td>
      <td>1.979848</td>
      <td>1.967973</td>
      <td>0.931721</td>
      <td>Sometimes</td>
      <td>Overweight_Level_II</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Ordinal encoding
for col in ['CAEC','CALC','NObeyesdad']:
    print(data[col].unique())
```

    ['Sometimes' 'Frequently' 'no' 'Always']
    ['Sometimes' 'no' 'Frequently']
    ['Overweight_Level_II' 'Normal_Weight' 'Insufficient_Weight'
     'Obesity_Type_III' 'Obesity_Type_II' 'Overweight_Level_I'
     'Obesity_Type_I']
    


```python
data['CAEC'] = data['CAEC'].map({'no':0,'Sometimes':1,'Frequently':2,'Always':3})
data['CALC'] = data['CALC'].map({'no':0,'Sometimes':1,'Frequently':2})
data['NObeyesdad'] = data['NObeyesdad'].map({'Insufficient_Weight':0,'Normal_Weight':1,'Overweight_Level_I':2,'Overweight_Level_II':3,'Obesity_Type_I':4,'Obesity_Type_II':5,'Obesity_Type_III':6})
data[['CAEC','CALC','NObeyesdad']].head()
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
      <th>CAEC</th>
      <th>CALC</th>
      <th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



The dataset has been cleaned and encoded, before moving on to modelling we will create a few new features to help provide more information to our predictive model.

## Feature Engineering
<a id='engineering'></a>


```python
# Feature engineering
```


```python
data_eng = data.copy()
data_eng[['Height','Weight']].describe() # Looks like metric unit
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
      <th>Height</th>
      <th>Weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20758.000000</td>
      <td>20758.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.700245</td>
      <td>87.887768</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.087312</td>
      <td>26.379443</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.450000</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.631856</td>
      <td>66.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.700000</td>
      <td>84.064875</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.762887</td>
      <td>111.600553</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.975663</td>
      <td>165.057269</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_eng['bmi'] = data_eng['Weight'] / (data_eng['Height'] ** 2)
data_eng[['Height','Weight','bmi']].head()
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
      <th>Height</th>
      <th>Weight</th>
      <th>bmi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.699998</td>
      <td>81.669950</td>
      <td>28.259565</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.560000</td>
      <td>57.000000</td>
      <td>23.422091</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.711460</td>
      <td>50.165754</td>
      <td>17.126706</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.710730</td>
      <td>131.274851</td>
      <td>44.855798</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.914186</td>
      <td>93.798055</td>
      <td>25.599151</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_eng[['NCP','FCVC']].describe()
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
      <th>NCP</th>
      <th>FCVC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20758.000000</td>
      <td>20758.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.761332</td>
      <td>2.445908</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.705375</td>
      <td>0.533218</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.393837</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_eng['veg_meal_ratio'] = data_eng['FCVC']/data_eng['NCP']
data_eng[['NCP','FCVC','veg_meal_ratio']].head()
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
      <th>NCP</th>
      <th>FCVC</th>
      <th>veg_meal_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.983297</td>
      <td>2.000000</td>
      <td>0.670399</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.411685</td>
      <td>1.880534</td>
      <td>1.332120</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.971472</td>
      <td>2.679664</td>
      <td>1.359220</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Modelling
<a id='modelling'></a>


```python
X_train = data_eng.drop('NObeyesdad',axis=1)
X.head()
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
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>...</th>
      <th>family_history_with_overweight_yes</th>
      <th>FAVC_yes</th>
      <th>SMOKE_yes</th>
      <th>SCC_yes</th>
      <th>MTRANS_Bike</th>
      <th>MTRANS_Motorbike</th>
      <th>MTRANS_Public_Transportation</th>
      <th>MTRANS_Walking</th>
      <th>bmi</th>
      <th>veg_meal_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.443011</td>
      <td>1.699998</td>
      <td>81.669950</td>
      <td>2.000000</td>
      <td>2.983297</td>
      <td>1</td>
      <td>2.763573</td>
      <td>0.000000</td>
      <td>0.976473</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>28.259565</td>
      <td>0.670399</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.000000</td>
      <td>1.560000</td>
      <td>57.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>23.422091</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18.000000</td>
      <td>1.711460</td>
      <td>50.165754</td>
      <td>1.880534</td>
      <td>1.411685</td>
      <td>1</td>
      <td>1.910378</td>
      <td>0.866045</td>
      <td>1.673584</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>17.126706</td>
      <td>1.332120</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.952737</td>
      <td>1.710730</td>
      <td>131.274851</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1</td>
      <td>1.674061</td>
      <td>1.467863</td>
      <td>0.780199</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>44.855798</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>31.641081</td>
      <td>1.914186</td>
      <td>93.798055</td>
      <td>2.679664</td>
      <td>1.971472</td>
      <td>1</td>
      <td>1.979848</td>
      <td>1.967973</td>
      <td>0.931721</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>25.599151</td>
      <td>1.359220</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
y_train = data_eng['NObeyesdad']
y.head()
```




    0    3
    1    1
    2    0
    3    6
    4    3
    Name: NObeyesdad, dtype: int64




```python
# Instantitate classifiers and their params
logreg = LogisticRegression(random_state = 47)
logreg_params = {'C': np.logspace(-4, 4, 6),
                 'solver': ['lbfgs','newton-cg','sag','saga'],
                 'multi_class': ['ovr','multinomial']
                }
rfc = RandomForestClassifier(random_state = 47)
rfc_params = {'n_estimators': [10,50,100,250],
              'min_samples_split': [2, 5, 10, 20]
             }
xgb = XGBClassifier(random_state = 47, device = 'gpu')
xgb_params = {'booster': ['gbtree','dart'],
              'eta': [0.01,0.3],
              'max_depth': [3,6,9],
              'lambda': [0.1,1]
             }
              
lgb = LGBMClassifier(random_state = 47)
lgb_params = {'max_bin': [10,69,150,255,400],
              'learning_rate': [ 0.01, 0.1],
              'num_leaves': [10,31]
             }

clfs = [
    ('Logistic Regression', logreg, logreg_params),
    ('Random Forest Classifier', rfc, rfc_params),
    ('XGBoost Classifier', xgb, xgb_params),
    ('LGBM Classifier', lgb, lgb_params)
]

scorers = {
    'accuracy_score': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score, average='micro')
}
```


```python
# Pipeline
results = []
for clf_name, clf, clf_params in clfs:
    gs = GridSearchCV(estimator=clf, 
                      param_grid=clf_params,
                      scoring=scorers,
                      refit='accuracy_score',
                      verbose=1
                     )
    pipeline = Pipeline(steps=[
        ('scaler', MinMaxScaler()),
        ('classifier', gs),
    ])
    pipeline.fit(X, y)
    result = [clf_name, gs.best_params_, gs.best_score_, gs.cv_results_['mean_test_f1_score'][gs.best_index_]]
    results.append(result)
result_df = pd.DataFrame(results, columns=['Name','Parameters','Accuracy','F1'])
result_df.head()
```

    Fitting 5 folds for each of 48 candidates, totalling 240 fits
    Fitting 5 folds for each of 16 candidates, totalling 80 fits
    Fitting 5 folds for each of 24 candidates, totalling 120 fits
    Fitting 5 folds for each of 20 candidates, totalling 100 fits
    




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
      <th>Accuracy</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Logistic Regression</td>
      <td>{'C': 10000.0, 'multi_class': 'multinomial', '...</td>
      <td>0.866462</td>
      <td>0.866462</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Random Forest Classifier</td>
      <td>{'min_samples_split': 5, 'n_estimators': 250}</td>
      <td>0.901532</td>
      <td>0.901532</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBoost Classifier</td>
      <td>{'booster': 'gbtree', 'eta': 0.3, 'lambda': 1,...</td>
      <td>0.906542</td>
      <td>0.906542</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LGBM Classifier</td>
      <td>{'learning_rate': 0.1, 'max_bin': 400, 'num_le...</td>
      <td>0.905723</td>
      <td>0.905723</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Process test data
test_df = pd.read_csv('test.csv')
ids = test_df['id']
test_df = test_df.drop('id', axis=1)
```


```python
# Nominal Encode
test_dummy_var = pd.get_dummies(test_df[dummy_cols], drop_first=True, dtype=int)
test_df = pd.concat([test_df, test_dummy_var], axis=1)
test_df = test_df.drop(dummy_cols,axis=1)

# Ordinal Encode
test_df['CAEC'] = test_df['CAEC'].map({'no':0,'Sometimes':1,'Frequently':2,'Always':3})
test_df['CALC'] = test_df['CALC'].map({'no':0,'Sometimes':1,'Frequently':2})
#test_df['NObeyesdad'] = test_df['NObeyesdad'].map({'Insufficient_Weight':0,'Normal_Weight':1,'Overweight_Level_I':2,'Overweight_Level_II':3,'Obesity_Type_I':4,'Obesity_Type_II':5,'Obesity_Type_III':6})

# Feature Engineering
test_df['bmi'] = test_df['Weight'] / (test_df['Height'] ** 2)
test_df['veg_meal_ratio'] = test_df['FCVC']/test_df['NCP']

# Feature Scaling (Fit on full training data)
scaler = MinMaxScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])
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
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CAEC</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
      <th>CALC</th>
      <th>...</th>
      <th>family_history_with_overweight_yes</th>
      <th>FAVC_yes</th>
      <th>SMOKE_yes</th>
      <th>SCC_yes</th>
      <th>MTRANS_Bike</th>
      <th>MTRANS_Motorbike</th>
      <th>MTRANS_Public_Transportation</th>
      <th>MTRANS_Walking</th>
      <th>bmi</th>
      <th>veg_meal_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26.899886</td>
      <td>1.848294</td>
      <td>120.644178</td>
      <td>2.938616</td>
      <td>3.000000</td>
      <td>1</td>
      <td>2.825629</td>
      <td>0.855400</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>35.315411</td>
      <td>0.979539</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21.000000</td>
      <td>1.600000</td>
      <td>66.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>25.781250</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.000000</td>
      <td>1.643355</td>
      <td>111.600553</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1</td>
      <td>2.621877</td>
      <td>0.000000</td>
      <td>0.250502</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>41.324115</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.979254</td>
      <td>1.553127</td>
      <td>103.669116</td>
      <td>2.000000</td>
      <td>2.977909</td>
      <td>1</td>
      <td>2.786417</td>
      <td>0.094851</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>42.976937</td>
      <td>0.671612</td>
    </tr>
    <tr>
      <th>4</th>
      <td>26.000000</td>
      <td>1.627396</td>
      <td>104.835346</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1</td>
      <td>2.653531</td>
      <td>0.000000</td>
      <td>0.741069</td>
      <td>1.0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>39.584143</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Get best classifier's parameters
print(*zip(result_df[result_df['Name'] == 'XGBoost Classifier']['Parameters']))
```

    ({'booster': 'gbtree', 'eta': 0.3, 'lambda': 1, 'max_depth': 3},)
    


```python
best_clf = XGBClassifier(booster='gbtree', eta=0.3, reg_lambda=1, max_depth=3, random_state=42)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(test_df)
```


```python
# Formatting and mapping predictions for kaggle submission
predictions = pd.concat([ids, pd.Series(y_pred)], axis = 1)
predictions.columns = 'ids','NObeyesdad'
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
      <th>ids</th>
      <th>NObeyesdad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20758</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20759</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20760</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20761</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20762</td>
      <td>6</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13835</th>
      <td>34593</td>
      <td>3</td>
    </tr>
    <tr>
      <th>13836</th>
      <td>34594</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13837</th>
      <td>34595</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13838</th>
      <td>34596</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13839</th>
      <td>34597</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>13840 rows × 2 columns</p>
</div>




```python
target_map = {'Insufficient_Weight':0,'Normal_Weight':1,'Overweight_Level_I':2,'Overweight_Level_II':3,'Obesity_Type_I':4,'Obesity_Type_II':5,'Obesity_Type_III':6}
target_unmap = {v: k for k, v in target_map.items()}
target_unmap
```




    {0: 'Insufficient_Weight',
     1: 'Normal_Weight',
     2: 'Overweight_Level_I',
     3: 'Overweight_Level_II',
     4: 'Obesity_Type_I',
     5: 'Obesity_Type_II',
     6: 'Obesity_Type_III'}




```python
data['NObeyesdad'] = data['NObeyesdad'].map(target_unmap)

```


```python
predictions.to_csv('predictions.csv')
```

<a id='conclusion'></a>
## 6. Conclusion
In this notebook we have explored a synthetic obesity dataset and trained a predictive model for a multi-class classification. The model was tuned using accuracy as an evaluation metric based on Kaggle competition rules, and managed to achieve a score of 0.90552 on the test set.
