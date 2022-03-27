import pandas as pd

class data_model:
    def __init__(self,metadatapath='data\\train.fea',timeseriespath='data\\croppedtimeseries.fea'):
        self.metadata = pd.read_feather(metadatapath)
        self.timeseries = pd.read_feather(timeseriespath)
    
    def datetime(self):
        return self.timeseries['datetime']

    def panels(self):
        return [int(x) for x in self.timeseries.columns[1:].tolist()]
    
    def index(self,i):
        i = int(i)
        row_metadata = self.metadata.loc[self.metadata['ss_id'] == i]
        row_series = self.timeseries[str(i)]
        return (row_metadata, row_series)
    
    