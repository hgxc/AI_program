# --coding:utf-8 --
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.svm import NuSVC
from sklearn.linear_model import Lasso,LogisticRegression
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
import xgboost as xgb

def data_processing(train,length):
    '''数据处理'''
    #打标签
    train.columns=['age','workclass','fnlwgt','educatio','edu_time','marry','occupation','relation','race','sex','gain','loss','hour_pre_week','nation','salary']

    y_train=train.pop('salary')#获取标签

    '''观察样本情况'''
    x=set(train['workclass'])
    y=set(train['occupation'])
    z=set(train['nation'])

    a=set(train['marry'])
    #print('workclass',x)
    #print('occup',y)
    #print('nation',z)
    #print('marry',a)
    em=train.isnull().sum()
    print(em[em>0].sort_values(ascending=False))

    print('workclass_count',Counter(train['workclass']))
    print('occupation_count',Counter(train['occupation']))
    print('nation_count',Counter(train['nation']))
    print('sex_count',Counter(train['sex']))
    print('y_count',Counter(y_train))

    '''数据处理'''
    #填充缺失数据
    cols=['workclass','occupation','nation']
    train['workclass'].fillna(' Private',inplace=True)
    train['nation'].fillna(' United-States',inplace=True)
    train['occupation'].fillna(' un-known',inplace=True)


    em1=train.isnull().sum()
    print(em1)



    #对数据进行增删改
    train = change_feature(train)
    print(train.shape)
    print(train.head())

    #大于50K为1，否则为0
    for i in range(length):
        if(y_train[i]==' >50K' or y_train[i]== ' >50K.'):
            y_train[i]=1
        else:
            y_train[i]=0
    n=set(y_train)
    print(n)
    print(Counter(y_train))

    '''特征编码'''
    dummied_data=pd.get_dummies(train)
    print(dummied_data)
    print(dummied_data.shape)
    print(y_train.head(10))

    #对数值型数据进行归一化

    #z-score 标准化
    #ss=StandardScaler()
    #std=ss.fit_transform(dummied_data)
    #print('std',std.shape)

    #o最小最大标准化
    mm=MinMaxScaler()
    std=mm.fit_transform(dummied_data)

    num_cols=dummied_data.columns[dummied_data.dtypes!='object']
    num_mean=dummied_data.loc[:,num_cols].mean()
    num_std=dummied_data.loc[:,num_cols].std()
    dummied_data.loc[:,num_cols]=(dummied_data.loc[:,num_cols]-num_mean)/num_std
    print(dummied_data.head())

    #choose_feature(dummied_data,y_train.astype('int'))

    #X=PCA_processing(std)
    return std,y_train

def change_feature(train):
    '''增删数据'''
    train['cost']=train['gain']-train['loss']
    train.pop('gain')
    train.pop('loss')
    #train.pop('occupation')
    train.pop('workclass')
    train.pop('fnlwgt')
    #train.pop('sex')
    train.pop('nation')
    #train.pop('race')
    return train

def choose_feature(x_train,y_train):
    '''特征选择'''
    lasso=Lasso(alpha=0.001)
    lasso.fit(x_train,y_train)
    FI_lasso = pd.DataFrame({"Feature Importance":lasso.coef_}, index=x_train.columns)
    FI_lasso[FI_lasso["Feature Importance"]!=0].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))
    plt.xticks(rotation=90)
    plt.show()

def PCA_processing(X):
    '''PCA处理'''
    pca=PCA(n_components=36)
    newX=pca.fit_transform(X)
    print('newX',newX)

    print(newX.shape)
    print(pca.explained_variance_ratio_)
    return newX

def create_con_matrix(y_true,y_pred):
    '''绘制混淆图'''
    label=[0,1]
    sns.set()
    f,ax=plt.subplots()
    pic=confusion_matrix(y_true,y_pred,label)
    print(pic)
    sns.heatmap(pic,annot=True,ax=ax)
    ax.set_title('confusion maxtrix')
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.show()




def logistic(x_train, x_test, y_train, y_test):
    '''逻辑回归'''
    model=LogisticRegression()
    model.fit(x_train,y_train.astype('int'))
    y_pred=model.predict(x_test)
    print(model.score(x_test,y_test.astype('int')))
    create_con_matrix(y_test.astype('int'),y_pred)

def de_tree(x_train, x_test, y_train, y_test):
    '''决策树'''
    dt=tree.DecisionTreeClassifier()
    dt.fit(x_train,y_train.astype('int'))
    y_pred=dt.predict(x_test)
    print(dt.score(x_test,y_test.astype('int')))
    create_con_matrix(y_test.astype('int'),y_pred)

def xg_boost(x_train, x_test, y_train, y_test):
    '''xgboost'''
    xgbr=xgb.XGBClassifier(n_estimators=140,subsample=0.8,colsample_bytree=0.8)
    xgbr.fit(x_train,y_train.astype('int'))
    y_pred=xgbr.predict(x_test)
    print(xgbr.score(x_test,y_test.astype('int')))
    create_con_matrix(y_test.astype('int'),y_pred)

def adb(x_train, x_test, y_train, y_test):
    '''adaboost'''
    adbt=AdaBoostClassifier()
    adbt.fit(x_train,y_train.astype('int'))
    y_pred=adbt.predict(x_test)
    print(adbt.score(x_test,y_test.astype('int')))
    create_con_matrix(y_test.astype('int'),y_pred)

if __name__=='__main__':
    train=pd.read_csv('C:\\Users\\xc\\Desktop\\ML\\adult_train.csv',header=None,na_values=" ?")
    print(train.shape)
    test=pd.read_csv('C:\\Users\\xc\\Desktop\\ML\\adult_test.csv',header=None,na_values=" ?")
    print(test.shape)
    train.columns=['age','workclass','fnlwgt','educatio','edu_time','marry','occupation','relation','race','sex','gain','loss','hour_pre_week','nation','salary']
    test.columns=['age','workclass','fnlwgt','educatio','edu_time','marry','occupation','relation','race','sex','gain','loss','hour_pre_week','nation','salary']
    data=pd.concat((train,test),keys=['age','workclass','fnlwgt','educatio','edu_time','marry','occupation','relation','race','sex','gain','loss','hour_pre_week','nation','salary'])

    train_data,train_label=data_processing(data,data.shape[0])
    print('label',set(train_label))

    x_train,x_test,y_train,y_test=train_test_split(train_data,train_label,test_size=1/3)



    print(x_train.shape)
    print(y_train)
    #使用SMOTE进行过采样
    #over_sample=SMOTE(random_state=520)
    #over_sample_X,over_sample_y=over_sample.fit_sample(x_train,y_train.astype('int'))
    #print(y_train.value_counts()/len((y_train)))
    #print(pd.Series(over_sample_y).value_counts()/len(over_sample_y))

    #训练模型
    #logistic(x_train, x_test, y_train, y_test)
    #de_tree(x_train, x_test, y_train, y_test)
    #xg_boost(x_train, x_test, y_train, y_test)
    adb(x_train, x_test, y_train, y_test)








