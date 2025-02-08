import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') 


np.random.seed(42)

small_size=100
medium_size=5000
large_size=20000

lags = [1,2,3,4,5,6]

noise_x_small = np.random.normal(loc=0,scale=1,size=small_size)
logging.info("Random noise generated for small series x")
noise_x_medium = np.random.normal(loc=0,scale=1,size=medium_size)
logging.info("Random noise generated for medium series x")
noise_x_large = np.random.normal(loc=0,scale=1,size=large_size)
logging.info("Random noise generated for large series x")

noise_y_small = np.random.normal(loc=0,scale=1,size=small_size)
logging.info("Random noise generated for small series y")
noise_y_medium = np.random.normal(loc=0,scale=1,size=medium_size)
logging.info("Random noise generated for medium series y")
noise_y_large = np.random.normal(loc=0,scale=1,size=large_size)
logging.info("Random noise generated for large series y")


def get_dataset(x_noise,y_noise,lags):
    df = pd.DataFrame({'x':x_noise})
    for lag in lags:
        y = np.roll(x_noise,lag)+y_noise
        df[f'y_{lag}'] = y 
    return df

df_small = get_dataset(noise_x_small,noise_y_small,lags)
logging.info("Generated dataset for benchmarking small series")
df_medium = get_dataset(noise_x_medium,noise_y_medium,lags)
logging.info("Generated dataset for benchmarking medium series")
df_large = get_dataset(noise_x_large,noise_y_large,lags)
logging.info("Generated dataset for benchmarking large series")

logging.info("Writing tables")
df_small.to_csv("../datasets/benchmark_small.csv",index=False)
df_medium.to_csv("../datasets/benchmark_medium.csv",index=False)
df_large.to_csv("../datasets/benchmark_large.csv",index=False)
logging.info("Finished Writing tables")