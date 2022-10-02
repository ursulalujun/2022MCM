import pandas as pd
def cal_macd_system(data,short_,long_,m):
    '''
    data是包含高开低收成交量的标准dataframe
    short_,long_,m分别是macd的三个参数
    返回值是包含原始数据和diff,dea,macd三个列的dataframe
    '''
    data['diff']=data['Value'].ewm(adjust=False,alpha=2/(short_+1),ignore_na=True).mean()-\
                data['Value'].ewm(adjust=False,alpha=2/(long_+1),ignore_na=True).mean()
    data['dea']=data['diff'].ewm(adjust=False,alpha=2/(m+1),ignore_na=True).mean()
    data['macd']=2*(data['diff']-data['dea'])
    data.to_excel('macd.xlsx')
    return data

df=pd.read_csv('BCHAIN-MKPRU.csv',na_values='NAN')
short_=12
long_=26
m=9
data=cal_macd_system(df,short_,long_,m)
print(data)
