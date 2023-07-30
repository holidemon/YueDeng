import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
data = pd.read_excel('D:\Survey on the Current Status of Digital Usage among University Students in Hunan Province\digital literacy--5.27.xlsx',sheet_name='Sheet1')
X = data.drop(labels='digital literacy',axis=1)
Y = data['digital literacy']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
n_estimators = [int(x) for x in range(1,300,2)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in range(1,20,1)]
min_sample_split = [2,5.10]
min_sample_leaf = [1,2,4]
random_state = [int(x) for x in range(10,100,2)]
random_grid = {
    'random_state':random_state,
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_sample_split,
    'min_samples_leaf':min_sample_leaf}
model = RandomForestRegressor()
rf_raddom= RandomizedSearchCV(estimator=model,param_distributions =random_grid,n_iter=300,scoring="neg_mean_absolute_error",cv=3,verbose =2,random_state=2,n_jobs=1)
rf_raddom.fit(x_train,y_train)
print(rf_raddom.best_params_)
