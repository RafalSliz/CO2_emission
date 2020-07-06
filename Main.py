import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import wget

#----------------- Downloading Data ----------------
#url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv'
#FuelConsumptionData = wget.download(url)

#---------------- Reading data ---------------------
df = pd.read_csv('FuelConsumptionCo2.csv')
#----- print all columns
pd.set_option('display.max_columns', None)
#----- take a look at the dataset
print(df.head())

#----- summarize the data
print(df.describe())

#----- select some features to explore
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

#----- plot each of these features
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show()
plt.clf()

#----- plot fuel consumption vs the emission, to see how linear is their relation
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
plt.clf()

#----- plot engine size vs the emission, to see how linear is their relation
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
plt.clf()

#----- plot cylinders vs the emission, to see how linear is their relation
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Cylinders")
plt.ylabel("Emission")
plt.show()
plt.clf()

#------------- Creating train and test dataset --------
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#------------ Simple regression model ----------------
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)
#------ the coefficients
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
#------ plot outputs
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
plt.clf()

#---------------- Evaluation --------------------------
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regr.predict(test_x)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )