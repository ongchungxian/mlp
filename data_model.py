import pandas as pd

class data_model:
    def __init__(self,metadatapath1='data\\train.fea',metadatapath2='data\\test.fea',timeseriespath='data\\croppedtimeseries.fea'):
        self.metadata1 = pd.read_feather(metadatapath1)
        self.metadata2 = pd.read_feather(metadatapath2)
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
    
    def train(self,its=1,length=64, pred_length=1, max_its=18):#generate a training dataset of the given length
        if(its>max_its):
            its=max_its
        x_metadata = []
        x_timeseries = []
        y = []
        for i in range(its):
            for j in range(170):
                x_metadata.append(self.metadata1.iloc[[j]][['ss_id','latitude_rounded','longitude_rounded','llsoacd','orientation','tilt','kwp']])
                station = str(self.metadata1.iloc[[j]]['ss_id'].iat[0])
                offset1 = 170*(length+pred_length)*i
                offset2 = j*(length+pred_length)
                full_offset = offset1+offset2
                time_chunk = []
                for k in range(length):
                    curr_index = full_offset+k
                    time_chunk.append(self.timeseries.iloc[[curr_index]][station].iat[0])
                x_timeseries.append(time_chunk)
                y.append(self.timeseries.iloc[[full_offset+length]][station].iat[0])
        return (x_metadata,x_timeseries, y)





    