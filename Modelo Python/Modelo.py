import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from matplotlib import pyplot as plt
# import matplotlib 
# import seaborn as sn
# import lazypredict
# from lazypredict.Supervised import LazyClassifier
# from lazypredict.Supervised import LazyRegressor
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import loguniform

df1 = pd.read_excel('Dataset.xlsx')

#Outlier Precio
dfoutlier = df1[df1['Precio(SMLV)'] <= 460]
#Outlier Baños
dfoutlier = dfoutlier[dfoutlier['Baños'] < 5]
#Outlier BHK
dfoutlier = dfoutlier[dfoutlier['BHK'] < 9]
#Outlier m^2
dfoutlier = dfoutlier[dfoutlier['m^2'] < 330]
#Eliminar Variables 
df2 = dfoutlier.drop('Precio(SMLV)',axis='columns')
df3 = df2.drop('Ubicacion',axis='columns')


# ------------------------------ IGNORAR ALGUNA COLUMNA (df3 defecto) ----------------------- 
# df5 = df3.drop('Estado',axis='columns')
# df4 = df3.drop('Estrato',axis='columns')
# df6 = df3.drop('Baños',axis='columns')
# df7 = df3.drop('m^2',axis='columns')
# df8 = df3.drop('BHK',axis='columns')
# df9 = df3.drop(['Baños'],axis='columns')
#####

X = df3
# ------------------------------ NO HACE FALTA NORMALIZAR -----------------------------------
# x_array = np.array(X['m^2'])
# normalized_arr = preprocessing.normalize([x_array])
# print(normalized_arr)
# X['m^2'] = normalized_arr.reshape(-1)
Y = dfoutlier['Precio(SMLV)']
#####

#--------------------------------- ENCONTRAR MEJOR RANDOM STATE --------------------------
# BestData = 0
# rs = 0
# for j in range(10000):  
#     X_train,X_test , y_train , y_test = train_test_split(X,Y,test_size=0.3,random_state=j)
#     lr_clf = LinearRegression()
#     lr_clf.fit(X_train.values,y_train)
#     aux = lr_clf.score(X_test.values,y_test)
#     if (BestData < aux):
#         BestData = aux
#         rs = j
# print(rs)
#####

#------------------------------------- ENTRENAMIENTO DATA SET -----------------------------
X_train,X_test , y_train , y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
lr_clf = GradientBoostingRegressor(n_estimators=100,max_leaf_nodes=10,learning_rate=0.111)
lr_clf.fit(X_train.values,y_train)
print(lr_clf.score(X_test.values,y_test))
# y_pred = lr_clf.predict(X_test)
# from sklearn import metrics
# print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
# Normalizada
# lr_clf.fit(X_train,y_train)
# print(lr_clf.score(X_test,y_test))
#####

#--------------------------- PREDECIR  ----------------------------------
def predict_price(estrato,m2,bhk,bath,estado):   

    a = [[estrato,m2,bhk,bath,estado]]
    return lr_clf.predict(a)[0]
print(predict_price(*[2,150,4,3,0]))
# print(predict_price(*[3,80,2,2,1]))
# print(predict_price(*[3,84,2,2,0]))
# print(predict_price(*[2,60,2,1,1]))
# print(predict_price(*[3,140,5,3,0]))
# print(predict_price(*[2,60,3,1,1]))
# print(predict_price(*[3,90,4,2,1]))
# print(predict_price(*[2,180,6,3,0]))
# print(predict_price(*[5,100,3,2,0]))
# print(predict_price(*[5,240,6,8,0]))
# print(predict_price(*[5,250,4,5,1]))
# print(predict_price(*[4,260,5,3,1]))
# print(predict_price(*[4,168,4,2,1]))
# print(predict_price(*[4,260,5,4,1]))
# print(predict_price(*[5,300,5,4,0]))
# print(predict_price(*[5,191,4,2,1]))
# print(predict_price(*[5,100,3,3,1]))
# print(predict_price(*[5,312,5,3,1]))
# print(predict_price(*[5,258,5,5,1]))
# print(predict_price(*[2,260,7,3,1]))
###

# #-------------------------- PROBAR ERROR -------------------------------
# for j in range (len(X)):
#     print(round(predict_price(*X.iloc[j].values)))
#####

#-------------------------- GRAFICAS DATA SET -------------------------
# corr_matrix = df2.corr()
# sn.heatmap(corr_matrix, annot=True)
# plt.show()
# def histogram(df):
#     matplotlib.rcParams['figure.figsize'] = (8,5)
#     upper_limit = df.mean() + 2*df.std()
#     lower_limit = df.mean() - 2*df.std()
#     # print('Limite Superior: Media + 2*std')
#     # print('')
#     # print(df.mean(),'+',2*df.std())
#     # print('')
#     # print('Limite Superior: ',upper_limit)
#     # print('Lower Limit: ',lower_limit)
#     # print(df[df > 459].count())
   
#     plt.hist(df,bins=40,rwidth=0.8)
#     plt.xlabel('Precio SMLV')
#     plt.ylabel('Cantidad')
#     plt.show()


# plt.boxplot(dfoutlier['m^2'])
# plt.show()
# plt.hist(dfoutlier['Estrato'],bins=40,rwidth=0.8)
# plt.show()
# def plot_scatter_chart_bhk(df,estrato):
#     bhk1 = df[ (df.BHK==1)]
#     bhk2 = df[ (df.BHK==2)]
#     bhk3 = df[ (df.BHK==3)]
#     bhk4 = df[ (df.BHK==4)]
#     bhk5 = df[ (df.BHK==5)]
#     bhk6 = df[ (df.BHK==6)]
#     bhk7 = df[ (df.BHK==7)]
#     bhk8 = df[ (df.BHK==8)]

#     matplotlib.rcParams['figure.figsize'] = (8,5)
#     plt.yticks(np.arange(Y.min()-7, Y.max(), 20))
#     plt.scatter(bhk1['m^2'],bhk1['Precio(SMLV)'],marker='1',color='red',label='1 BHK', s=50)
#     plt.scatter(bhk2['m^2'],bhk2['Precio(SMLV)'],color='red',label='2 BHK', s=50)
#     plt.scatter(bhk3['m^2'],bhk3['Precio(SMLV)'],marker='+', color='green',label='3 BHK', s=50)
#     plt.scatter(bhk4['m^2'],bhk4['Precio(SMLV)'],marker='*', color='green',label='4 BHK', s=50)
#     plt.scatter(bhk5['m^2'],bhk5['Precio(SMLV)'],marker='P', color='blue',label='5 BHK', s=50)
#     plt.scatter(bhk6['m^2'],bhk6['Precio(SMLV)'],marker='p', color='blue',label='6 BHK', s=50)
#     plt.scatter(bhk7['m^2'],bhk7['Precio(SMLV)'],marker='h', color='black',label='7 BHK', s=50)
#     plt.scatter(bhk8['m^2'],bhk8['Precio(SMLV)'],marker='+', color='yellow',label='8 BHK', s=50)

#     plt.xlabel("m^2")
#     plt.ylabel("Precio SMLV")
#     plt.title(estrato)
#     plt.legend()
#     plt.show()
# plot_scatter_chart_bhk(dfoutlier,'BHK')
# def plot_scatter_chart_banos(df,estrato):

#     bhk2 = df[ (df['Baños']==1)]
#     bhk3 = df[ (df['Baños']==2)]
#     bhk4 = df[ (df['Baños']==3)]
#     bhk5 = df[ (df['Baños']==4)]


#     matplotlib.rcParams['figure.figsize'] = (8,5)
#     plt.yticks(np.arange(Y.min()-7, Y.max(), 20))
#     plt.scatter(bhk2['m^2'],bhk2['Precio(SMLV)'],color='red',label='Baños 1', s=50)
#     plt.scatter(bhk3['m^2'],bhk3['Precio(SMLV)'],marker='+', color='red',label='Baños 2', s=50)
#     plt.scatter(bhk4['m^2'],bhk4['Precio(SMLV)'],marker='*', color='green',label='Baños 3', s=50)
#     plt.scatter(bhk5['m^2'],bhk5['Precio(SMLV)'],marker='P', color='blue',label='Baños 4', s=50)
#     plt.xlabel("m^2")
#     plt.ylabel("Precio SMLV")
#     plt.title(estrato)
#     plt.legend()
#     plt.show()
# plot_scatter_chart_banos(dfoutlier,'Baños')
# def plot_scatter_chart_estado(df,estrato):

#     bhk2 = df[(df['Estado']==0)]
#     bhk3 = df[(df['Estado']==1)]
#     matplotlib.rcParams['figure.figsize'] = (8,5)
#     plt.yticks(np.arange(Y.min()-7, Y.max(), 20))
#     plt.scatter(bhk2['m^2'],bhk2['Precio(SMLV)'],color='blue',label='Excelente', s=50)
#     plt.scatter(bhk3['m^2'],bhk3['Precio(SMLV)'],marker='+', color='red',label='Aceptable', s=50)
#     plt.xlabel("m^2")
#     plt.ylabel("Precio SMLV")
#     plt.title(estrato)
#     plt.legend()
#     plt.show()
# plot_scatter_chart_estado(dfoutlier,'Estado')
# def plot_scatter_chart_estrato(df,estrato):

#     bhk2 = df[(df['Estrato']==2)]
#     bhk3 = df[(df['Estrato']==3)]
#     bhk4 = df[(df['Estrato']==4)]
#     bhk5 = df[(df['Estrato']==5)]
#     matplotlib.rcParams['figure.figsize'] = (8,5)
#     plt.yticks(np.arange(Y.min()-7, Y.max(), 20))
#     plt.scatter(bhk2['m^2'],bhk2['Precio(SMLV)'],color='Red',label='2', s=50)
#     plt.scatter(bhk3['m^2'],bhk3['Precio(SMLV)'],marker='+', color='green',label='3', s=50)
#     plt.scatter(bhk4['m^2'],bhk4['Precio(SMLV)'],marker='P', color='blue',label='4', s=50)
#     plt.scatter(bhk5['m^2'],bhk5['Precio(SMLV)'],marker='p', color='blue',label='5', s=50)
#     plt.xlabel("m^2")
#     plt.ylabel("Precio SMLV")
#     plt.title(estrato)
#     plt.legend()
#     plt.show()
# plot_scatter_chart_estrato(dfoutlier,'Estrato')


######k



#------------------------ Cross Val Score ------------------------
# from sklearn.model_selection import ShuffleSplit
# from sklearn.model_selection import cross_val_score

# cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# print(cross_val_score(LinearRegression(), X, Y, cv=cv))
#####

#------------------------ Comparacion Modelos ------------------------


# def find_best_model_using_gridsearchcv(X,y):
#     algos = {

#         'RandomForestRegressor' : {
#             'model': RandomForestRegressor(),
#             'params': {
#                 "max_features": [1, 2, 3, 5, None],
#                 "max_leaf_nodes": [10, 100, 1000, None],
#                 "min_samples_leaf": [1, 2, 5, 10, 20, 50, 100],
#             }
#         },
#         'GradientBoostingRegressor': {
#             'model': GradientBoostingRegressor(),
#             'params': {
#                 "n_estimators": [1, 2, 5, 10, 20, 50, 100, 200, 500],
#                 "max_leaf_nodes": [2, 5, 10, 20, 50, 100],
#                 "learning_rate": [0.095,0.092,0.05,0.01,0.1,0.125,0.11,0.12,0.115,0.111],
#             }
#         },        
#         'Ridge': {
#             'model': Ridge(),
#             'params': {
#                 'alpha':[0.0001, 0.001,0.01, 0.1, 1, 10]
#             }
#         },
#         'Lasso': {
#             'model': Lasso(),
#             'params': {
#                 'alpha':[0.0001, 0.001,0.01, 0.1, 1, 10]
#             }
#         }
        
#     }
#     scores = []
#     cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     print(cv)
#     for algo_name, config in algos.items():
#         gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
#         gs.fit(X,y)
#         scores.append({
#             'model': algo_name,
#             'best_score': gs.best_score_,
#             'best_params': gs.best_params_
#         })
#     pd.DataFrame(scores,columns=['model','best_score','best_params']).to_excel("output.xlsx")  
#     return pd.DataFrame(scores,columns=['model','best_score','best_params'])

# print(find_best_model_using_gridsearchcv(X,Y))

#------------------------ Pickle Dump ------------------------
# import pickle
# with open('ML_Model.pickle','wb') as f:
#     pickle.dump(lr_clf,f)
