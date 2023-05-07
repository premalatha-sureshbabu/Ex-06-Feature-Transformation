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
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()

```

# Output
## DATA
![11](https://user-images.githubusercontent.com/120620842/236667477-0e18d722-e945-4c6b-be27-10300a180731.png)

![12](https://user-images.githubusercontent.com/120620842/236667514-95dec1b7-4f3a-4457-a361-d36496323a96.png)

![13](https://user-images.githubusercontent.com/120620842/236667526-40994bb3-22ba-4782-8c51-494404a1618e.png)

![14](https://user-images.githubusercontent.com/120620842/236667547-53afcda0-bff3-4075-bf17-027dcb1e201b.png)

![15](https://user-images.githubusercontent.com/120620842/236667555-921c9db6-471f-4a31-b8ae-6b29194da49d.png)

## Before Transformation

![16](https://user-images.githubusercontent.com/120620842/236667590-d8404af4-80b2-405f-bafe-897ecacafaf0.png)

![17](https://user-images.githubusercontent.com/120620842/236667602-f207cf2c-d51b-4629-90ea-8b06b1febd81.png)

![18](https://user-images.githubusercontent.com/120620842/236667610-a91a8ce4-8bd1-46f8-a92d-73de1b396ddb.png)

![19](https://user-images.githubusercontent.com/120620842/236667623-807d366f-df04-4516-9865-c495c4c3cb16.png)

## Log Transformation

![20](https://user-images.githubusercontent.com/120620842/236667667-4689f5dd-31f3-4df9-b5f2-c7eb4d508c3e.png)

## Reciprocal Transformation

![21](https://user-images.githubusercontent.com/120620842/236667687-109d4d2e-7a61-4d95-ada3-55e7914f3aab.png)

## Square root Transformation

![22](https://user-images.githubusercontent.com/120620842/236667710-79b95ea5-8d70-421d-9acd-93f6b34e7493.png)

![23](https://user-images.githubusercontent.com/120620842/236667724-5b5efec4-0aa2-4643-b7fe-c92e4393a2b3.png)

## Power Transformation

![24](https://user-images.githubusercontent.com/120620842/236667747-71b82bfd-f08f-4c00-821a-7a8480b4dcb7.png)

## Quantile Transformation

![25](https://user-images.githubusercontent.com/120620842/236667764-95d9490e-a306-4700-a441-1120fd0b90fb.png)


# RESULT:
Thus the Feature Transformation for the given datasets had been executed successfully.
