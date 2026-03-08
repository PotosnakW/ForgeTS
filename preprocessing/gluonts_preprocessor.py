import pandas as pd
import numpy as np

def gluonts_to_long_dataframe(dataset_name="m4_yearly", split='train'):
    """
    Convert GluonTS dataset to long-format pandas DataFrame
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset (e.g., 'm4_yearly', 'm4_quarterly')
    split : str
        'train' or 'test'
    
    Returns:
    --------
    pd.DataFrame with columns: ['unique_id', 'ds', 'y']
    """
    from gluonts.dataset.repository.datasets import get_dataset
    dataset = get_dataset(dataset_name)
    data = dataset.train if split == 'train' else dataset.test
    
    all_series = []
    for idx, entry in enumerate(data):
        item_id = entry.get('item_id', entry.get('id', f'series_{idx}'))

        start = entry['start']
        target = entry['target']

        date_range = pd.date_range(
            start=start.to_timestamp(), 
            periods=len(target), 
            freq=start.freq
        )
        series_df = pd.DataFrame({
            'unique_id': item_id,
            'ds': date_range,
            'y': target
        })
        all_series.append(series_df)
    
    df_long = pd.concat(all_series, ignore_index=True)
    
    return df_long