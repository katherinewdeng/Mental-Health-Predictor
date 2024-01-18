import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_regression 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

scaler = StandardScaler()  
df = pd.read_csv(r"C:\Users\princ\OneDrive\Desktop\prokect\datasets\diabetes_012_health_indicators_BRFSS2015.csv")

df.MentHlth[df['MentHlth'] <= 14] = 0
df.MentHlth[df['MentHlth'] > 14] = 1

x = df[["Diabetes_012", "HighBP", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "NoDocbcCost", "GenHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]]
y = df["MentHlth"]

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

clf = MLPClassifier(hidden_layer_sizes=(30,15), activation = 'tanh',max_iter=50, alpha=0.001,
                     solver='adam', learning_rate= 'adaptive', verbose=3,tol=0.000000001)
'''
param_grid = {
    'hidden_layer_sizes': [(150,100,50), (120,80,40), (100,50,30)],
    'max_iter': [50, 100],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.001, 0.01, 0.5],
    'learning_rate': ['constant','adaptive'],
}
grid = GridSearchCV(clf, param_grid, n_jobs= -1, cv=5)
grid.fit(x_train, y_train)
print(grid.best_params_) 

grid_predictions = grid.predict(x_test) 
test_set_rsquared = grid.score(x_test, y_test)
test_set_rmse = np.sqrt(mean_squared_error(y_test, grid_predictions))
'''

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print('Training set score: {:.4f}'.format(clf.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(clf.score(x_test, y_test)))

mse =mean_squared_error(y_test, y_pred)
print('Mean Squared Error : '+ str(mse))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error : '+ str(rmse))

'''
df_results = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})

df_results['error'] = df_results['Actual'] - df_results['Predicted']
print(df_results.head(60))

print(test_set_rmse)
print(test_set_rsquared)
plt.plot(clf.loss_curve_)
plt.title("Loss Curve", fontsize=14)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

plt.scatter(y_pred, y_train, color="green")
plt.show()

'''

#print(accuracy_score(y_test, y_pred))

cm1 = confusion_matrix(y_test,y_pred)
labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
#sns.heatmap(cm1, annot=labels, fmt='', cmap='Blues')
sns.heatmap(cm1/np.sum(cm1), annot=True, 
            fmt='.2%', cmap='Blues')
cm1
plt.show()
