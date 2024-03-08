import pandas as pd
from tqdm.notebook import tqdm

def read_chunks(file, cols=None, city=None, chunk_size=500):
    '''
    Read dataset in chunks
    '''
    df = pd.read_json(path_or_buf=file, chunksize=chunk_size, lines=True)
    chunk_list = []
    for chunk in tqdm(df, desc=file):
        if city:
            chunk = chunk[chunk['city'] == city]
        if cols is None:
            chunk_list.append(chunk)
        else:
            chunk_list.append(chunk[cols])
    return pd.concat(chunk_list, ignore_index=True, join='outer', axis=0)

def load_filtered_data(city):
    '''
    Load filtered data from a city
    '''
    file = ['business', 'checkin', 'review', 'tip', 'user']
    data = {}
    for f in file:
        data[f] = pd.read_csv(f'filtered_cities/{city}_{f}.csv')
    return data
