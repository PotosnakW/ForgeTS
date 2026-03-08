import pandas as pd
import numpy as np
from gift_eval.data import Dataset

def preprocess_gifteval_dataset_for_neuralforecast(
        dataset_name, 
        dataset_type, 
        ids=None, 
        to_univariate=False,
        term='short'
    ):
    '''
    inputs:
        - dataset_name: str, name of dataset.
        - dataset_type: str=['train','test'], Specify dataset.
        - ids: list of ids to keep in dataset (generally used for dataset_type='test'). 
    return: preprocessed Dataframe with 'unique_id', 'ds', and 'y' columns.
    '''
    dataset = Dataset(
        name=dataset_name, 
        term=term, 
        to_univariate=to_univariate
    )
    
    if dataset_type == 'train':
        data_list = list(dataset.validation_dataset)
    elif dataset_type == 'test':
        data_list = [i[1] for i in dataset.test_data]
    else:
        raise ValueError("dataset_type must be 'train' or 'test'")
    
    df = pd.DataFrame(data_list)
    
    # known series with all nans
    PROBLEMATIC_SERIES = {
        'electricity': ['MT_178'],
        'bitbrains_fast_storage': ['fastStorage_552']
    }
    
    for dataset_key, series_ids in PROBLEMATIC_SERIES.items():
        if dataset_key in dataset.name:
            df = df[~df.item_id.isin(series_ids)]
    
    n_targets = [i.shape for i in df.target.values]
    has_multiple_targets = any(len(shape) == 2 for shape in n_targets)
    
    if has_multiple_targets:
        print('more than one target')
        dfe_list = []
        for n, target in enumerate(n_targets):
            dfe = pd.DataFrame(df.iloc[n]).T.explode("target")
            dfe_n_series = dfe.shape[0]
            dfe['item_id'] = [f'{dfe.item_id.values[0]}_{j}' for j in range(dfe_n_series)]
            dfe_list.append(dfe)
        dfe_all = pd.concat(dfe_list, axis=0, ignore_index=True)
    else:
        print('one target')
        dfe_all = df
    
    # expand data to get stacked unique_id values
    df_expanded = dfe_all.explode("target")
    df_expanded.reset_index(inplace=True, drop=True)
    
    metadata = dfe_all.groupby('item_id')[['start', 'freq']].first()
    for uid in metadata.index:
        mask = df_expanded.item_id == uid
        start = metadata.loc[uid, 'start'].to_timestamp()
        freq = metadata.loc[uid, 'freq']
        periods = mask.sum()
        df_expanded.loc[mask, 'ds'] = pd.date_range(start=start, periods=periods, freq=freq)
    
    df_expanded.drop(columns=['start', 'freq'], inplace=True)
    df_expanded.rename(columns={'target': 'y', 'item_id':'unique_id'}, inplace=True)
    df_expanded = df_expanded[['unique_id', 'ds', 'y']]
    df_expanded.reset_index(drop=True, inplace=True)
    
    df_expanded['available_mask'] = 0
    one_idxs = np.where(df_expanded["y"].notnull())[0]
    df_expanded.loc[one_idxs, "available_mask"] = 1
    df_expanded.y = df_expanded.groupby('unique_id')['y'].ffill().fillna(0)
    
    if (dataset_type == 'test') and (ids is not None):
        df_expanded = df_expanded[df_expanded['unique_id'].isin(ids)]
    
    print('n_series length:', df_expanded.shape[0]/df_expanded.unique_id.unique().shape[0])
    
    return df_expanded
