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

![12](https://user-images.githubusercontent.com/120620842/233551353-00496881-b503-4185-adbe-c0f472272359.png)

![13](https://user-images.githubusercontent.com/120620842/233551390-38c18bfc-1df8-4017-a14b-7cff3fed2f91.png)

![14](https://user-images.githubusercontent.com/120620842/233551413-73f9bce1-4204-4fd2-a9ee-216074959824.png)

![15](https://user-images.githubusercontent.com/120620842/233551432-b808496b-1c15-4595-bee7-d4e55cdb668b.png)

## Log Transformation
![16](https://user-images.githubusercontent.com/120620842/233551448-e5c6fec6-14da-41b3-b54c-e842e0430a76.png)

## Reciprocal Transformation
![17](https://user-images.githubusercontent.com/120620842/233551487-54ec03d7-8fe7-452a-8474-7d1e0a4f22d7.png)

## SquareRoot Transformation
![18](https://user-images.githubusercontent.com/120620842/233551518-3e2570ad-ddd3-42e1-ac84-022f1930e07f.png)

## Power Transformation
![19](https://user-images.githubusercontent.com/120620842/233551542-6ecc9a87-6be0-4fd4-b505-c73d2122c818.png)

![20](https://user-images.githubusercontent.com/120620842/233551567-f5f660f4-e6d2-453e-8007-401d5a1d6b86.png)

## Quantile Transformation
![21](https://user-images.githubusercontent.com/120620842/233551609-045b1764-3866-4a3b-97b2-76074337b122.png)

# RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully.
