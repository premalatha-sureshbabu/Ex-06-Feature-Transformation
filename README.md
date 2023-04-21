# Ex-06-Feature-Transformation

# AIM:
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
## STEP 1
Read the given Data

## STEP 2
Clean the Data Set using Data Cleaning Process

## STEP 3
Apply Feature Transformation techniques to all the features of the data set

## STEP 4
Save the data to the file

# CODE:

```

Name :S.Prema Latha
Register Number : 212222230112
Feature Transformation - Data_to_Transform.csv


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()
df1 = df.copy()
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)
sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()

```

# Output
## Feature Transformation - Data_to_Transform.csv
![11](https://user-images.githubusercontent.com/120620842/233551168-7ac84ca1-ac4c-4c88-8ee1-166324eb52e4.png)

![Screenshot 2023-04-21 112713](https://user-images.githubusercontent.com/120620842/233552096-6a8ed9d5-141d-428d-b649-f424d4c3ba79.png)

![Screenshot 2023-04-21 112724](https://user-images.githubusercontent.com/120620842/233552164-492de1f9-576d-419f-8d29-61edf61e3ea1.png)

![14](https://user-images.githubusercontent.com/120620842/233551413-73f9bce1-4204-4fd2-a9ee-216074959824.png)

![15](https://user-images.githubusercontent.com/120620842/233551432-b808496b-1c15-4595-bee7-d4e55cdb668b.png)

## Log Transformation
![16](https://user-images.githubusercontent.com/120620842/233551448-e5c6fec6-14da-41b3-b54c-e842e0430a76.png)

## Reciprocal Transformation
![Screenshot 2023-04-21 113056](https://user-images.githubusercontent.com/120620842/233552689-d51143c1-7ae5-459f-bdae-6d1f969d3994.png)


## SquareRoot Transformation
![Screenshot 2023-04-21 113109](https://user-images.githubusercontent.com/120620842/233552729-1fe28766-2fc7-4048-b170-94fc019030a6.png)


## Power Transformation
![Screenshot 2023-04-21 113125](https://user-images.githubusercontent.com/120620842/233552770-2bef50d8-fca6-4e6c-aa14-aadd8bd0b08b.png)

![20](https://user-images.githubusercontent.com/120620842/233551567-f5f660f4-e6d2-453e-8007-401d5a1d6b86.png)

## Quantile Transformation
![Screenshot 2023-04-21 113143](https://user-images.githubusercontent.com/120620842/233552814-138211ee-8109-4e2e-b284-5747c159f4ee.png)


# RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully.
