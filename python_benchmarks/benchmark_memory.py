from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd 
from memory_profiler import memory_usage
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 

def profile_memory(data:pd.DataFrame,lag_data:int,max_lag:int):
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
    


def create_results(data:pd.DataFrame,tag:str,lag_data:int,max_lag:int,num_iters=10):
    mem = [] 
    for _ in range(num_iters):
        m = memory_usage((profile_memory,(data,lag_data,max_lag,),{}))
        mem.append(max(m))
    result = pd.DataFrame()
    result['memory_usage(MiB)']=mem 
    result['tag']=tag
    result['avg_memory_usage(MiB)']=pd.Series(mem).mean()
    result['std_memory_usage(MiB)']=pd.Series(mem).std()
    return result 
if __name__=='__main__':
    
    df_small = pd.read_csv('../datasets/benchmark_small.csv')
    df_medium = pd.read_csv('../datasets/benchmark_medium.csv')
    df_large = pd.read_csv('../datasets/benchmark_large.csv')

    logging.info("Running benchmarks for small dataset")
    results_small_mem = create_results(df_small,'small_data_lag_1_max_lag_5',1,5)
    logging.info("Running benchmarks for medium dataset") 
    results_medium_mem = create_results(df_medium,'medium_data_lag_1_max_lag_5',1,5)
    logging.info("Running benchmarks for large dataset")
    results_large_mem = create_results(df_large,'large_data_lag_1_max_lag_5',1,5)
    

    results_small_mem.to_csv("../datasets/memory_small.csv",index=False)
    results_medium_mem.to_csv("../datasets/memory_medium.csv",index=False)
    results_large_mem.to_csv("../datasets/memory_large.csv",index=False)