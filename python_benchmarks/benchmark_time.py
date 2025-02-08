from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd 
import time
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 


df_small = pd.read_csv('../datasets/benchmark_small.csv')
df_medium = pd.read_csv('../datasets/benchmark_medium.csv')
df_large = pd.read_csv('../datasets/benchmark_large.csv')

def profile_time(data:pd.DataFrame,lag_data:int,max_lag:int):
    start_time = time.time() 
    x = data['x']
    if lag_data == 1:
        y = data['y_1']
    if lag_data == 2:
        y = data['y_2']
    if lag_data==3:
        y = data['y_3']
    if lag_data==4:
        y= data['y_4']
    if lag_data==5:
        y=data['y_5']
    if lag_data==6:
        y=data['y_6']
    if y is None:
        raise Exception('Provide the lag in data to be tested')
    data = pd.DataFrame({'x':x.values,'y':y.values})
    _ = grangercausalitytests(data, maxlag=max_lag, verbose=True)
    end_time = time.time()
    duration = end_time-start_time
    return duration
    
def create_results(data:pd.DataFrame,tag:str,lag_data:int,max_lag:int,num_iters=10):
    durations = [] 
    for _ in range(num_iters):
        d = profile_time(data,lag_data,max_lag)
        durations.append(d)
    result = pd.DataFrame()
    result['run_times']=durations 
    result['tag']=tag
    result['avg_time']=pd.Series(durations).mean()
    result['std_time']=pd.Series(durations).std()
    return result 

results_small_time = create_results(df_small,'small_data_lag_1_max_lag_5',1,5)
logging.info("Running benchmarks for small dataset")
results_medium_time = create_results(df_medium,'medium_data_lag_1_max_lag_5',1,5)
logging.info("Running benchmarks for medium dataset") 
results_large_time = create_results(df_large,'large_data_lag_1_max_lag_5',1,5)
logging.info("Running benchmarks for large dataset")

results_small_time.to_csv("../datasets/timing_small.csv",index=False)
results_medium_time.to_csv("../datasets/timing_medium.csv",index=False)
results_large_time.to_csv("../datasets/timing_large.csv",index=False)