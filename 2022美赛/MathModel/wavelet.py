import pandas as pd
import matplotlib.pyplot as plt
import pywt
import math
import numpy as np

df=pd.read_csv('LBMA-GOLD.csv',na_values='NAN')
df=df.dropna(how='any')
data=df['USD (PM)'].tolist()
#create wavelet object and define parameters
w=pywt.Wavelet('db8')#选用Daubechies8小波
maxlev=pywt.dwt_max_level(len(data),w.dec_len)
print("maximum level is"+str(maxlev))
threshold=0.3  #Threshold for filtering

#Decompose into wavelet components,to the level selected:
coffs=pywt.wavedec(data,'db8',level=maxlev) #将信号进行小波分解

for i in range(1,len(coffs)):
    coffs[i]=pywt.threshold(coffs[i],threshold*max(coffs[i]))

datarec=pywt.waverec(coffs,'db8')#将信号进行小波重构
datarec=datarec[:1255]
t=range(1,len(data)+1)

plt.figure()
plt.subplot(3,1,1)
plt.plot(t, data)
plt.xlabel('time (s)')
plt.ylabel('Gold daily prices(dollars)')
plt.title("Raw data")
plt.subplot(3, 1, 2)
plt.plot(t, datarec)
plt.xlabel('time (s)')
plt.ylabel('Gold daily prices (dollars)')
plt.title("De-noised data using wavelet techniques")
plt.subplot(3, 1, 3)
plt.plot(t,data-datarec)
plt.xlabel('time (s)')
plt.ylabel('noise (dollars)')
plt.tight_layout()
plt.show()