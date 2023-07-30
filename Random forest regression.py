import numpy as np
import pandas as pd
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.tree import export_graphviz
data = pd.read_excel('D:\Survey on the Current Status of Digital Usage among University Students in Hunan Province\digital literacy --5.27.xlsx',sheet_name='Sheet1')
X = data.drop(labels='digital literacy ',axis=1)
Y = data['digital literacy ']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=10)
mode_fit = RandomForestRegressor(n_estimators=117,max_depth=15,max_features='sqrt',random_state=68,min_samples_split=2,min_samples_leaf=2)
mode_fit.fit(x_train,y_train)
prediction = mode_fit.predict(x_test)
predict = mode_fit.predict(x_train)
score = r2_score(y_test,prediction)
mse = mean_squared_error(y_test,prediction)
mse1 = r2_score(y_train,predict)
mae = mean_absolute_error(y_test,prediction)
print(f'拟合mse{mse},拟合mae{mae}')
importtance = mode_fit.feature_importances_
data.columns = ['digital literacy ','School Level','Monthly Family Income','Highest Education Level of Parents','Father’s Occupation','Mother’s Occupation','Family Location',
                'Family Internet access time','Digital Facilities','Software Resources',
                'Digital Courses','Digital Competition Atmosphere','Usage Frequency','Number of Commonly Used Software',
                'Usage Time','Purpose of Usage']
feet_labels = data.columns[1:]
indices = np.argsort(importtance)[::-1]
for f in range(x_train.shape[1]):
    print("%2d)%-*s %f" %(f+1,30,feet_labels[indices[f]],importtance[indices[f]]))

tree = mode_fit.estimators_[0]
dot_data = export_graphviz(tree, out_file=None,
                         feature_names=x_train.columns,
                         class_names=['0','1'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("DTtree.pdf")
