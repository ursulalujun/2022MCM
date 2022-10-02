import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from openpyxl import load_workbook
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.impute import SimpleImputer
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,LSTM
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,mean_absolute_percentage_error
from torch import dropout

def save_to_excel(data,excelName,sheetName):  
    writer = pd.ExcelWriter(excelName)
    book = load_workbook(excelName)
    writer.book = book
    data.to_excel(writer, sheet_name=sheetName)
    writer.save()

def outliner(df):
    clf = LOF(n_neighbors=10)
    y=np.array(df['USD (PM)']).reshape(1265,1)
    res = clf.fit_predict(y) # Label is 1 for an inlier and -1 for an outlier
    outlier=[]
    y=y.astype(np.float) 
    for i in range(1461):
        if res[i]==-1:
            outlier.append(i)
            print(clf.negative_outlier_factor_[i])
    """         if clf.negative_outlier_factor_[i]>-20:
                y[i]=np.nan
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(y)
    y=imp.transform(y)
    for i in range(1461):
        df.iloc[i,2]=y[i] """

def sliding_window(sequence, len_train, len_pre, fearure):
    X,y=[],[]
    length=len(sequence)
    for i in range(0,length-30):
        pre_start=i+len_train
        feature_one=fearure.iloc[pre_start-1,:].tolist()
        X_one=sequence[i:i+len_train]+feature_one
        y_one=sequence[pre_start:pre_start+len_pre]
        X.append(X_one)
        y.append(y_one)    
    return X,y

def split_data(X0,y0):
    index1=758
    index2=1412
    X=X0[:index2,:]
    valX=X0[index2:,:]
    y=y0[:index2,:]
    valY=y0[index2:,:]
    return X, y, valX, valY

def scaler(X):
    column_headers=X.columns 
    min_max_scaler = preprocessing.MinMaxScaler()    
    scaled_X=pd.DataFrame(min_max_scaler.fit_transform(X))
    scaled_X.columns=column_headers
    return scaled_X

# fit an LSTM model
def fit_model(X,y,valX, valY):
    timesteps=1
    X=X.reshape(X.shape[0],timesteps,X.shape[1])
    valX=valX.reshape(valX.shape[0],timesteps,valX.shape[1])
    # define model
    model = Sequential()
    model.add(LSTM(256, input_shape=(timesteps,X.shape[2])))
    model.add(Dense(5, activation='linear'))   
    # compile model
    model.compile(loss='mse', optimizer='adam')    
    # fit model
    history = model.fit(X, y, batch_size=32, epochs=40, shuffle=False, verbose=0)
    # evaluate model
    loss = model.evaluate(valX, valY, verbose=0)
    return loss

def lstm(X_train,X_test,y_train,y_test):
    timesteps=1
    print(X_train.shape,X_test.shape,y_test.shape)
    X_train=X_train.reshape(X_train.shape[0],timesteps,X_train.shape[1])
    X_test=X_test.reshape(X_test.shape[0],timesteps,X_test.shape[1])
    model = Sequential()
    # LSTM需要3维输入(batch_size, timesteps, input_dim)
    model.add(LSTM(128,activation='relu', input_shape=(timesteps, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(5))
    model.compile(optimizer='adam', loss='mse')
    # model.summary()
    # fit network
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    df1=pd.DataFrame(y_train_pred)
    df2=pd.DataFrame( y_test_pred)
    df1=df1*63554.44
    df2=df2*63554.44
    """ min_max_scaler = preprocessing.MinMaxScaler()   
    df1 = min_max_scaler.inverse_transform(df1)
    df1 = min_max_scaler.inverse_transform(df1) """
    # print(df1)
    df1.to_excel('y_train_pred_b.xlsx')
    df2.to_excel('y_test_pred_b.xlsx')
    return y_train_pred,y_test_pred

def evaluate_model(y_train,y_train_pred,y_test,y_pred):
    df1=pd.DataFrame(y_train_pred)
    df2=pd.DataFrame( y_pred)
    index=pd.DataFrame({})
    index2=pd.DataFrame({})
    index['y_index']= range(y_test.shape[0])
    index2['y']=range(0,df2.shape[0])
    plt.plot(index2['y'] , df2[0], color="red", label='predict', linewidth=1)
    plt.plot(index['y_index'] , y_test, color="blue", label='price', linewidth=1)
    plt.xticks((np.arange(0, 250, step=50)))#设置坐标的刻度和名称 
    plt.ylim([0, 1])      
    plt.legend() 
    plt.show()  
    index=pd.DataFrame({})
    index2=pd.DataFrame({})
    index['y_index']= range(y_train.shape[0])
    index2['y']=range(0,df1.shape[0])
    plt.plot(index2['y'] , df1[0], color="blue", label='price', linewidth=1)
    plt.plot(index['y_index'] , y_train, color="red",label='predict',linewidth=1)
    plt.xticks((np.arange(0, 760, step=50)))#设置坐标的刻度和名称 
    plt.ylim([0, 1])       
    plt.legend() 
    plt.show()

def pre_plot(y_train,y_train_pred,y_test,y_pred):
    index=pd.DataFrame({})
    index['y_index']= range(y_test.shape[0])
    plt.plot(index['y_index'] , y_test, color="blue", label='price', linewidth=1)
    len=y_pred.shape[0]
    for i in range(0,len,8):
        r=pd.DataFrame({})
        r['index']=range(i,i+5)
        plt.plot(r , y_pred[i]-0.02, color="red",linewidth=1)
    print(y_pred)
    plt.xticks((np.arange(0, 250, step=50)))#设置坐标的刻度和名称     
    plt.ylim([0, 1])   
    plt.legend() 
    plt.show()  
    index=pd.DataFrame({})
    index['y_index']= range(y_train.shape[0])
    plt.plot(index['y_index'] , y_train, color="blue", label='price', linewidth=1)
    len=y_train_pred.shape[0]
    for i in range(0,len,8):
        r=pd.DataFrame({})
        r['index']=range(i,i+5)
        plt.plot(r, y_train_pred[i], color="red",linewidth=1)
    plt.xticks((np.arange(0, 760, step=50)))#设置坐标的刻度和名称    
    plt.ylim([0, 1])    
    plt.legend() 
    plt.show()

# 9.28开始
df=pd.read_csv('BCHAIN-MKPRU.csv',na_values='NAN').iloc[12:,1:]
df=scaler(df)
df2=pd.read_csv('bitcoin.csv').iloc[:,1:]
df2=scaler(df2)
df=df.fillna(method='ffill')
df2=df2.fillna(method='ffill')
len_train=15
len_pre=5
print(df.shape,df2.shape)
# economic_feature()
X,y=sliding_window(df['Value'].tolist(), len_train, len_pre, df2)
X=np.array(X)
y=np.array(y)
X, y, valX, valY=split_data(X,y)
len1=y.shape[0]
len2=valY.shape[0]
#y1_plot=df['USD (PM)'].iloc[len_train:len_train+len1]
#y2_plot=df['USD (PM)'].iloc[len_train+len1:len_train+len1+len2]
y1_plot=df['Value'].iloc[len_train:1412]
y2_plot=df['Value'].iloc[1412+len_train:]
print(X.shape,y.shape)
y_train_pred,y_test_pred=lstm(X,valX,y,valY)
# evaluate_model(y1_plot,y_train_pred,y2_plot,y_test_pred)
pre_plot(y1_plot,y_train_pred,y2_plot,y_test_pred)
