import codecs
import pandas as pd
import json

class DataLoader:
    """
    loading different datasets!
    """
    def __init__(self):
        pass

    def read_texts(self,path, split=False):
        """
        reading plain text document, 
        split is for spliting text into seprated lines(default is False)
        """
        if split:
            return open(path,"r").read().split('\n')
        else:
            return open(path,"r").read()
        
    def load_json(self, path): 
        "loading json files"
        return json.load(open(path, 'r'))

    def read_df(self, path, df_type="csv", sep = None, encoding = None, names=None):
        """loading pandas pkl and csv files"""
        
        config = {"sep":sep, "encoding":encoding, "df_type":df_type, "path":path, "names":names}
        
        return self._df_handler(config)

    def _df_handler(self, config):
        df = pd.read_pickle if config['df_type'] == 'pkl' else pd.read_csv
        if config['sep'] != None and config['encoding'] != None and config['names'] != None:
            return df(config['path'], sep=config['sep'], encoding=config['encoding'], names=config['names'])
        elif config['encoding'] != None and config['names'] != None:
            return df(config['path'], encoding=config['encoding'], names=config['names'])
        elif config['sep'] != None:
            return df(config['path'], sep=config['sep'])
        else:
            return df(config['path'])
