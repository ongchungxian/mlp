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

    def train(self, length=64,pred_length=1):
        its = 200185/(length+pred_length)
        x_ms = []
        x_ts = []
        y_s = []
        for i in range(its):
            x_m, x_t, y = self.train_func(i, length, pred_length)
            x_ms.append(x_m)
            x_ts.append(x_t)
            y_s.append(y)
        return (x_ms, x_ts, y)
    
    def train_func(self,i,length=64, pred_length=1):#generate a training dataset of the given length
        x_metadata = []
        x_timeseries = []
        y = []

        for j in range(170):
            station = str(self.metadata1.iloc[[j]]['ss_id'].iat[0])
            full_offset = (length+pred_length)*i
            time_chunk = []
            metadatum1 = self.metadata1.iloc[[j]][['ss_id','latitude_rounded','longitude_rounded','llsoacd','orientation','tilt','kwp']].reset_index().drop('index',1)
            metadatum2 = self.timeseries.iloc[[full_offset]]['datetime'].reset_index().drop('index',1)
            metadatum = pd.concat([metadatum1, metadatum2], axis=1)
            x_metadata.append(metadatum)
            for k in range(length):
                curr_index = full_offset+k
                time_chunk.append(self.timeseries.iloc[[curr_index]][station].iat[0])
            x_timeseries.append(time_chunk)
            for k in range(pred_length):
                y.append(self.timeseries.iloc[[full_offset+length+k]][station].iat[0])
        return (x_metadata,x_timeseries, y)

    def test(self, length=64,pred_length=1):
        its = 200185/(length+pred_length)
        x_ms = []
        x_ts = []
        y_s = []
        for i in range(its):
            x_m, x_t, y = self.test_func(i, length, pred_length)
            x_ms.append(x_m)
            x_ts.append(x_t)
            y_s.append(y)
        return (x_ms, x_ts, y)


    def test_func(self,i,length, pred_length):#generate a test dataset of the given length
        x_metadata = []
        x_timeseries = []
        y = []
        for j in range(85):
            station = str(self.metadata1.iloc[[j]]['ss_id'].iat[0])
            full_offset = (length+pred_length)*i
            time_chunk = []
            metadatum1 = self.metadata1.iloc[[j]][['ss_id','latitude_rounded','longitude_rounded','llsoacd','orientation','tilt','kwp']]
            metadatum2 = self.timeseries.iloc[[full_offset]]['datetime']
            metadatum = pd.concat([metadatum1, metadatum2], axis=1)
            x_metadata.append(metadatum)
            for k in range(length):
                curr_index = full_offset+k
                time_chunk.append(self.timeseries.iloc[[curr_index]][station].iat[0])
            x_timeseries.append(time_chunk)
            for k in range(pred_length):
                y.append(self.timeseries.iloc[[full_offset+length+k]][station].iat[0])
        return (x_metadata,x_timeseries, y)




    