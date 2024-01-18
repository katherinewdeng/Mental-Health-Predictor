import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import imblearn as imb

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from pandas import DataFrame
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from matplotlib.ticker import PercentFormatter

scaler = StandardScaler()  
OGdata = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\prokect\datasets\diabetes_012_health_indicators_BRFSS2015.csv")
data = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\prokect\datasets\diabetes_012_health_indicators_BRFSS2015.csv")


data["Diabetes_012"] = data["Diabetes_012"].astype(int)
data["HighBP"] = data["HighBP"].astype(int)
data["HighChol"] = data["HighChol"].astype(int)
data["CholCheck"] = data["CholCheck"].astype(int)
data["BMI"] = data["BMI"].astype(int)
data["Smoker"] = data["Smoker"].astype(int)
data["Stroke"] = data["Stroke"].astype(int)
data["HeartDiseaseorAttack"] = data["HeartDiseaseorAttack"].astype(int)
data["PhysActivity"] = data["PhysActivity"].astype(int)
data["Fruits"] = data["Fruits"].astype(int) 
data["Veggies"] = data["Veggies"].astype(int)
data["HvyAlcoholConsump"] = data["HvyAlcoholConsump"].astype(int)
data["AnyHealthcare"] = data["AnyHealthcare"].astype(int)
data["NoDocbcCost"] = data["NoDocbcCost"].astype(int)
data["GenHlth"] = data["GenHlth"].astype(int)
data["MentHlth"] = data["MentHlth"].astype(int)
data["PhysHlth"] = data["PhysHlth"].astype(int)
data["DiffWalk"] = data["DiffWalk"].astype(int)
data["Sex"] = data["Sex"].astype(int)
data["Age"] = data["Age"].astype(int)
data["Education"] = data["Education"].astype(int)
data["Income"] =data["Income"].astype(int)

data.MentHlth[data['MentHlth'] <= 14] = 0
data.MentHlth[data['MentHlth'] > 14] = 1


#data["Diabetes_binary_str"]= data["Diabetes_012"].replace({0:"NonDiabetic",1:"Diabetic"})
data2 = data.copy() 

data2.Age[data2['Age'] == 1] = "18 to 24"
data2.Age[data2['Age'] == 2] = "25 to 29"
data2.Age[data2['Age'] == 3] = "30 to 34"
data2.Age[data2['Age'] == 4] = "35 to 39"
data2.Age[data2['Age'] == 5] = "40 to 44"
data2.Age[data2['Age'] == 6] = "45 to 49"
data2.Age[data2['Age'] == 7] = "50 to 54"
data2.Age[data2['Age'] == 8] = '55 to 59'
data2.Age[data2['Age'] == 9] = '60 to 64'
data2.Age[data2['Age'] == 10] = '65 to 69'
data2.Age[data2['Age'] == 11] = '70 to 74'
data2.Age[data2['Age'] == 12] = '75 to 79'
data2.Age[data2['Age'] == 13] = '80 or older'

data2.Diabetes_012[data2['Diabetes_012'] == 0] = 'No Diabetes'
data2.Diabetes_012[data2['Diabetes_012'] == 1] = 'Diabetes'
data2.Diabetes_012[data2['Diabetes_012'] == 2] = 'Diabetes'

data2.HighBP[data2['HighBP'] == 0] = 'No High'
data2.HighBP[data2['HighBP'] == 1] = 'High BP'

data2.HighChol[data2['HighChol'] == 0] = 'No High Cholesterol'
data2.HighChol[data2['HighChol'] == 1] = 'High Cholesterol'

data2.CholCheck[data2['CholCheck'] == 0] = 'No Cholesterol Check in 5 Years'
data2.CholCheck[data2['CholCheck'] == 1] = 'Cholesterol Check in 5 Years'

data2.Smoker[data2['Smoker'] == 0] = 'No'
data2.Smoker[data2['Smoker'] == 1] = 'Yes'

data2.Stroke[data2['Stroke'] == 0] = 'No'
data2.Stroke[data2['Stroke'] == 1] = 'Yes'

data2.HeartDiseaseorAttack[data2['HeartDiseaseorAttack'] == 0] = 'No'
data2.HeartDiseaseorAttack[data2['HeartDiseaseorAttack'] == 1] = 'Yes'

data2.PhysActivity[data2['PhysActivity'] == 0] = 'No'
data2.PhysActivity[data2['PhysActivity'] == 1] = 'Yes'

data2.Fruits[data2['Fruits'] == 0] = 'No'
data2.Fruits[data2['Fruits'] == 1] = 'Yes'

data2.Veggies[data2['Veggies'] == 0] = 'No'
data2.Veggies[data2['Veggies'] == 1] = 'Yes'

data2.HvyAlcoholConsump[data2['HvyAlcoholConsump'] == 0] = 'No'
data2.HvyAlcoholConsump[data2['HvyAlcoholConsump'] == 1] = 'Yes'

data2.AnyHealthcare[data2['AnyHealthcare'] == 0] = 'No'
data2.AnyHealthcare[data2['AnyHealthcare'] == 1] = 'Yes'

data2.NoDocbcCost[data2['NoDocbcCost'] == 0] = 'No'
data2.NoDocbcCost[data2['NoDocbcCost'] == 1] = 'Yes'

data2.GenHlth[data2['GenHlth'] == 5] = 'Excellent'
data2.GenHlth[data2['GenHlth'] == 4] = 'Very Good'
data2.GenHlth[data2['GenHlth'] == 3] = 'Good'
data2.GenHlth[data2['GenHlth'] == 2] = 'Fair'
data2.GenHlth[data2['GenHlth'] == 1] = 'Poor'

data2.DiffWalk[data2['DiffWalk'] == 0] = 'No'
data2.DiffWalk[data2['DiffWalk'] == 1] = 'Yes'

data2.Sex[data2['Sex'] == 0] = 'Female'
data2.Sex[data2['Sex'] == 1] = 'Male'

data2.Education[data2['Education'] == 1] = 'Never Attended School'
data2.Education[data2['Education'] == 2] = 'Elementary'
data2.Education[data2['Education'] == 3] = 'Junior High School'
data2.Education[data2['Education'] == 4] = 'Senior High School'
data2.Education[data2['Education'] == 5] = 'Undergraduate Degree'
data2.Education[data2['Education'] == 6] = 'Magister'

data2.Income[data2['Income'] == 1] = 'Less Than $10,000'
data2.Income[data2['Income'] == 2] = 'Less Than $10,000'
data2.Income[data2['Income'] == 3] = 'Less Than $10,000'
data2.Income[data2['Income'] == 4] = 'Less Than $10,000'
data2.Income[data2['Income'] == 5] = 'Less Than $35,000'
data2.Income[data2['Income'] == 6] = 'Less Than $35,000'
data2.Income[data2['Income'] == 7] = 'Less Than $35,000'
data2.Income[data2['Income'] == 8] = '$75,000 or More'



data.duplicated().sum()
data.drop_duplicates(inplace = True) 
data.duplicated().sum()
data.shape

#historgram
'''
data.hist(figsize=(20,15))
'''

#correlation btwn age and mental health
'''
sns.boxplot(x = 'MentHlth', y = 'Age', data = data)
plt.title('Age vs MentHlth')
plt.show()
'''

#correlation btwn age and mental health
'''
pd.crosstab(data2.Age,data2.MentHlth).plot(kind="bar",figsize=(20,6))
plt.title('Mental Health Frequency for Ages')
plt.xlabel('Age')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()
'''

'''
pd.crosstab(data2.DiffWalk,data2.MentHlth).plot(kind="bar",figsize=(10,6))
plt.title('Mental Health Decline vs Physical Health Decline')
plt.xlabel('Physical Health (the bigger the number the worse)')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()
'''


#education compared to mental health CURRENTLY NOT WORKING PROPERLY
'''
sns.displot(data.Education[data.MentHlth <= 14], color="y", label="Less Mental Health Issues" )
sns.displot(data.Education[data.MentHlth > 14], color="m", label="More Mental Health Issues" )
plt.title("Relation between Education and Mental Health")
plt.legend()
plt.show()
'''

#BMI and mental health CURRENTLY NOT WORKING PROPERLY
'''
plt.figure(figsize=(25, 15))
sns.countplot(data.BMI[data.MentHlth <= 14], color="r", label="Less Mental Health Issues")
sns.countplot(data.BMI[data.MentHlth > 14], color="g", label="More Mental Health Issues")
plt.title("Relation b/w BMI and Mental Health")
plt.legend()
plt.show()
'''

#Diabetes and Mental Health correlation CURRENTLY NOT WORKING EITHER ):
''''
pd.crosstab(data.MentHlth,data.Diabetes_binary_str).plot(kind="bar",figsize=(30,12),color=['#1CA53B', '#FFA500' ])
plt.title('Diabetes Disease Frequency for MentHlth')
plt.xlabel('MentHlth')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.show()
'''
'''
data.drop('MentHlth', axis=1).corrwith(data.MentHlth).plot(kind='bar', grid=True, figsize=(20, 8), title="Correlation with Mental Health!",color="Purple")
plt.show()
'''

X = OGdata.iloc[:,0:]
Y = OGdata.iloc[:,15]

BestFeatures = SelectKBest(score_func=chi2, k=10)
fit = BestFeatures.fit(X,Y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

#concatenating two dataframes for better visualization
f_Scores = pd.concat([df_columns,df_scores],axis=1)               # feature scores
f_Scores.columns = ['Feature','Score']

print(f_Scores.nlargest(16,'Score'))

colomns = ["Fruits" , "Veggies" , "HighChol" , "CholCheck" , "AnyHealthcare" , "HvyAlcoholConsump"]
data.drop(colomns , axis= 1 ,inplace=True)


x = data[["Diabetes_012", "HighBP", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "NoDocbcCost", "GenHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]]
y = data["MentHlth"]

print(y.value_counts())

'''
plt.figure(figsize = (20,10))
sns.heatmap(data.corr('pearson'),annot=True , cmap ='YlOrRd' )
plt.title("correlation of feature")
'''

sm = SMOTE(random_state = 2) 
x_sm,y_sm= sm.fit_resample(x,y)
print(y_sm.shape)
print(x_sm.shape)
print(y_sm.value_counts()) 


x_train, x_test, y_train, y_test = train_test_split(x_sm,y_sm, test_size= 0.30, random_state=27)
scaler.fit(x_train)  
x_train = scaler.fit_transform(x_train)  
# apply same transformation to test data
x_test = scaler.fit_transform(x_test)   


lg = LogisticRegression(max_iter = 5000)
lg.fit(x_train , y_train)

y_pred=lg.predict(x_test)
print('Training set score: {:.4f}'.format(lg.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(lg.score(x_test, y_test)))

mse =mean_squared_error(y_test, y_pred)
print('Mean Squared Error : '+ str(mse))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error : '+ str(rmse))


cm1 = confusion_matrix(y_test,y_pred)
labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cm1/np.sum(cm1), annot=True, 
            fmt='.2%', cmap='Blues')
cm1
plt.show()


dt = tree.DecisionTreeClassifier(max_depth = 50)
dt.fit(x_train , y_train)
y_pred=dt.predict(x_test)

print('Training set score: {:.4f}'.format(dt.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(dt.score(x_test, y_test)))

mse =mean_squared_error(y_test, y_pred)
print('Mean Squared Error : '+ str(mse))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error : '+ str(rmse))

cm1 = confusion_matrix(y_test,y_pred)
labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
#sns.heatmap(cm1, annot=labels, fmt='', cmap='Blues')
sns.heatmap(cm1/np.sum(cm1), annot=True, 
            fmt='.2%', cmap='Blues')
cm1
plt.show()


'''
cm1 = confusion_matrix(y_test,y_pred)
f = sns.heatmap(cm1, fmt='d')
plt.show()
'''
