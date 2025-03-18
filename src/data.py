import os
import warnings
import pandas as pd
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient
from influxdb_client.client.warnings import MissingPivotFunction
from loguru import logger

load_dotenv()

warnings.simplefilter('ignore', MissingPivotFunction)

#logger.debug([os.getenv(v) for v in ['INFLUX_URL', 'INFLUX_TOKEN', 'INFLUX_ORG', 'INFLUX_BUCKET']])

class DataframeManager:
  def __init__(
    self,
    measurements    = ['DS18B20', 'SI7021', 'RPi Zero 2W'],
    labels          = ['core', 'room', 'cpu'],
    dimension       = 'temperature',
    field           = 'value',
    time_range      = '7d',
    agg_window      = '5m',
    agg_function    = 'mean',
    resample_window = '5min',
    bucket          = os.getenv('INFLUX_BUCKET')
  ):
    self.measurements    = measurements
    self.labels          = labels
    self.dimension       = dimension
    self.field           = field
    self.bucket          = bucket
    self.time_range      = time_range
    self.agg_window      = agg_window
    self.agg_function    = agg_function
    self.resample_window = resample_window
    
    self.client = InfluxDBClient(
      url    = os.getenv('INFLUX_URL'),
      token  = os.getenv('INFLUX_TOKEN'),
      org    = os.getenv('INFLUX_ORG'),
      bucket = bucket
    )
    self.query_api = self.client.query_api()
    
  def create_query_string(self, measurement:str):
    return f'''
    from(bucket: "{self.bucket}")
        |> range(start: -{self.time_range})
        |> filter(fn: (r) => r["_measurement"] == "{measurement}")
        |> filter(fn: (r) => r["dimension"] == "{self.dimension}")
        |> filter(fn: (r) => r["_field"] == "{self.field}")
        |> aggregateWindow(every: {self.agg_window}, fn: {self.agg_function}, createEmpty: false)
    '''
    
  def get_dataframe(self, measurement, label):
    query_string = self.create_query_string(measurement)
    logger.debug(query_string)
    query_df = self.query_api.query_data_frame(query=query_string)
    query_df = self.process_dataframe(query_df, label)
    return query_df
    
  def process_dataframe(self, df, label):
    df = df[['_time', '_value']].rename(columns={'_value': label})
    df.set_index('_time', inplace=True)
    df = df.resample(self.resample_window).mean()
    return df
    
  def get_merged_dataframe(self, fill_missing=True):
    logger.debug(zip(self.measurements, self.labels))
    dfs = [self.get_dataframe(m, l) for m, l in zip(self.measurements, self.labels)]
    logger.debug(dfs)
    df  = pd.concat(dfs, axis=1)
    if fill_missing:
      df = df.fillna(method='ffill').fillna(method='bfill')
    return df
  
  
if __name__ == '__main__':
  manager = DataframeManager()
  df = manager.get_merged_dataframe()
  
  print(df.head())
  
  