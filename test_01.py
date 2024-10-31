#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:35:22 2024

@author: mons
"""
#  https://www.geeksforgeeks.org/random-forest-hyperparameter-tuning-in-python/


from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 

data = pd.read_csv( 
	"https://raw.githubusercontent.com/lucifertrj/"
	"100DaysOfML/main/Day14%3A%20Logistic_Regression"
	"_Metric_and_practice/heart_disease.csv") 

print(data.head(7) )

print(data['target'].value_counts() )


X = data.drop("target", axis=1) 
y = data['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42) 

print(X_train.shape, X_test.shape )
model = RandomForestClassifier() 
model.fit(X_train, y_train) 

# predict the mode 
y_pred = model.predict(X_test) 

# performance evaluatio metrics 
print(classification_report(y_pred, y_test)) 

param_grid = { 
	'n_estimators': [25, 50, 100, 150], 
	'max_features': ['sqrt', 'log2', None], 
	'max_depth': [3, 6, 9], 
	'max_leaf_nodes': [3, 6, 9], 
} 

grid_search = GridSearchCV(RandomForestClassifier(), 
						param_grid=param_grid) 
grid_search.fit(X_train, y_train) 
print(grid_search.best_estimator_) 


model_grid = RandomForestClassifier(max_depth=9, 
									max_features="log2", 
									max_leaf_nodes=9, 
									n_estimators=25) 
model_grid.fit(X_train, y_train) 
y_pred_grid = model.predict(X_test) 
print(classification_report(y_pred_grid, y_test)) 


