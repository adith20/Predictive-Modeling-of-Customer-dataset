import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("venv//dataset_testing_01172022_PNW.csv")  # Dataset
                                                                                 # Attribute selection - Correlation Matrix
corr = sns.clustermap(data.corr(), annot=True, cmap="YlGnBu",figsize=(20, 10),linewidths=0.5) 
print(corr)

data.loc[data["CAS_NUM"] <100, ["CAS_NUM"]] = 0

pd.options.mode.chained_assignment = None

data = data.dropna()   # data cleaning
print(data.shape)
                                                                                # converting risk score value to 5 categories for improved prediction
                                                                                
data.loc[(data["CAS_NUM"] >=100) & (data["CAS_NUM"] <150), "CAS_NUM"] = 1
data.loc[(data["CAS_NUM"] >=150) & (data["CAS_NUM"] <200), "CAS_NUM"] = 2
data.loc[(data["CAS_NUM"] >=200) & (data["CAS_NUM"] <250), "CAS_NUM"] = 3
data.loc[(data["CAS_NUM"] >=250) & (data["CAS_NUM"] <300), "CAS_NUM"] = 4
data.loc[(data["CAS_NUM"] >=300) & (data["CAS_NUM"] <330), "CAS_NUM"] = 5

from sklearn.model_selection import train_test_split

X = data[['CA_ID','TENURE','PAYMENT_PLAN_FLAG','Num_DQ_6M','NUM_NOPAY_3M']].values     # With attribute selection method, we downsampled to 5
Y = data[['NUM_NOPAY_6M','CAS_NUM']].values                                            # attributes. Out of which Naive Bayes gives highest
print(X)                                                                               # accuracies with 2 attributes that are greatly influencing
print(Y)                                                                               # the class label - 'Num_DQ_6M' & 'NUM_NOPAY_3M'

from sklearn.naive_bayes import GaussianNB 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)    # split into training(80%)
                                                                            # and testing(20%)
model = GaussianNB()                                       # Model 1 for 1st prediction (No. of non-payments in 6 months) - Gaussian Naive Bayes
model.fit(X_train[:,3:], Y_train[:,0])

print(X_test.shape)
Y_pred = model.predict(X_test[:,3:])
print(Y_pred)

from sklearn.metrics import accuracy_score, classification_report
import plotly.graph_objects as go

accuracy = accuracy_score(Y_test[:,0], Y_pred)*100  # generating accuracy for the model 1
print('Accuracy is ',accuracy,'%')
metrics = classification_report(Y_test[:,0], Y_pred)     # generating confusion matrix for the model 1
print('Classification metrics is \n ',metrics)

                                                              # Tabular data for actual and predicted values
fig = go.Figure(data=[go.Table(header=dict(values=['CA_ID', 'Actual(No. of non-payments in 6 months)', 'Predicted(No. of non-payments in 6 months)']),
                    cells=dict(values=[X_test[:,0], Y_test[:,0], Y_pred]))
                        ])
fig.show()

model1 = GaussianNB()                                # Model 2 for 2nd prediction (Risk category level- to fall in debt)
model1.fit(X_train[:,1:2], Y_train[:,1])
print(Y_test[:,1])

Y_pred1 = model1.predict(X_test[:,1:2])
print(Y_pred1)

accuracy1 = accuracy_score(Y_test[:,1], Y_pred1)*100
print('Accuracy is ',accuracy1,'%')
                                                    # Tabular data for actual and predicted values
fig = go.Figure(data=[go.Table(header=dict(values=['CA_ID', 'Actual(Risk category level 0-5)', 'Predicted(Risk category level 0-5)']),
                    cells=dict(values=[X_test[:,0], Y_test[:,1], Y_pred1]))
                        ]),
fig.show()

#Sample test data to draw inferences from our predictions

index = int(input('Enter an index in the range (0 to 19784) to interpret predictions for a sample customer from our test data - '))
print("Customer ID - ",X_test[index,0])
print("Predicted number of Non-payments for the customer in next 6 Months - ",Y_pred[index])
print("Predicted Risk category Level for the customer - ",Y_pred1[index])

risk = (Y_pred1[index]/5)*100      # risk % = predcited level/total risk levels
if (Y_pred[index] >3):
  if (Y_pred1[index] >= 3):  
    print('We predict the customer to have',Y_pred[index],'non-payments in next 6 months & has',risk,'% chances of falling in debt in future')
    if (X_test[index,2] == 0):
      print('We suggest customer to be enrolled in a payment plan')
    else:
      print('Customer already enrolled in a payment plan. But we suggest to make it more feasible since customer has greater no. of non-payments & higher risk category')

  else:
    print('We predict the customer to have',Y_pred[index],'non-payments in next 6 months, but has only',risk,'% chances of falling in debt in future ')
    print('We suggest customer can be enrolled in a short-term payment plan')

else:
  if (Y_pred1[index] >= 3): 
    print('We predict the customer to have',Y_pred[index],'non-payments in next 6 months, but has',risk,'% chances of falling in debt in future')
    if (X_test[index,2] == 0):
      print('We suggest customer to be enrolled in a payment plan in the future')
    else:
      print('Customer already enrolled in a payment plan')
  else:
    print('We predcit the customer to have',Y_pred[index],'non-payments in next 6 months, and has only',risk,'% chances of falling in debt in future ')
    if (X_test[index,2] == 1):
      print('We suggest customer need not be enrolled in a payment plan')
    else:
      print('Customer not enrolled in a payment plan')