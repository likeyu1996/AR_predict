import tushare as tsh
import datetime
from pandas import DataFrame
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARMA

code_list=['000001','399001','399005','399006','000016','000300','000905']
code_name=['上证综指','深证成指','中小板指','创业板指','上证50','沪深300','中证500']
def get_index_data(code_list=code_list):
    n=len(code_list)
    data_list=[0 for i in range(n)]
    now=str(datetime.date.today())
    k=0
    for code_id in code_list:
        table=tsh.get_k_data(code=code_id,start='2016-01-04',end='2017-05-11',index=True,ktype='D')
        data=np.array(table['close'])
        data_list[k]=np.log([data[i]/data[i-1] for i in range(1,len(data))])
        k+=1
    data_array=np.array(data_list).reshape(n,-1)
    data_frame=DataFrame(data_array,index=code_list).T
    return data_frame
def LBQ_test(n):
    r,q,p = sm.tsa.acf(n, qstat=True)
    data = np.c_[range(1,41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    # print(table.set_index('lag'))
    return table
def adf_test(li,autolag='AIC'):
    adftest = ts.adfuller(li, autolag='%s' %autolag)
    adf_res = pd.Series(adftest[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    adf_res['Autolag']=autolag
    for key, value in adftest[4].items():
        adf_res['Critical Value (%s)' % key] = value
    return adf_res
# def ar_predict():
index_data=get_index_data()
# print(index_data)
# sns.plt.plot(index_data)
# sns.plt.show()
n_diff=12
for code_id in code_list:
    return_array=np.array(index_data['%s' %code_id])
    return_array_old=return_array[:-n_diff]
    return_array_new=return_array[-n_diff:]
    # LBQ_result=LBQ_test(return_array_old).set_index('lag')
    predict_list=[0 for i in range(n_diff)]

    for k in range(len(return_array_new)):
        return_array_old=np.append(return_array_old,return_array_new[k])
        adf_aic=adf_test(return_array_old)
        # adf_bic=adf_test(return_array_old,autolag='BIC')
        lag=int(adf_aic['Lags Used'])
        print('lag:',lag)
        model=ARMA(return_array_old,order=(lag,0))
        result_arma = model.fit(disp=-1,method='css')
        end_num=len(return_array_old)+1
        predict=result_arma.predict(end=end_num)
        zero_part=np.array([0.0 for i in range(lag)])
        true_predict=np.concatenate((zero_part,predict))
        predict_list[k]=true_predict[-1]

        # sns.plt.plot(true_predict)
        # sns.plt.plot(return_array)
        # sns.plt.plot(return_array_old,'Y')
        # sns.plt.show()
        # print(result_arma.arparams)
        # print(result_arma.params)
        # print(result_arma.pvalues)

    print(np.array(predict_list))
    print(return_array_new)
    sns.plt.plot(predict_list,label='predict')
    sns.plt.plot(return_array_new,label='real')
    sns.plt.ylim([-0.03,0.03])
    sns.plt.legend(loc="best")
    sns.plt.title("AR Predict %s" % code_id)
    sns.plt.show()
