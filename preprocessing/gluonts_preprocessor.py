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
        'train', 'test', or 'train_test'
    Returns:
    --------
    pd.DataFrame with columns: ['unique_id', 'ds', 'y']
    'train_test' also includes a 'split' column marking train/test rows.
    """
    from gluonts.dataset.repository.datasets import get_dataset
    dataset = get_dataset(dataset_name)

    if split == 'train_test':
        train_df = _load_split(dataset, 'train')
        test_df  = _load_split(dataset, 'test')
        df = pd.concat([train_df, test_df], ignore_index=True)
        df = df.drop_duplicates(subset=['unique_id', 'ds'], keep='last')
        df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
        return df

    elif split in ('train', 'test'):
        return _load_split(dataset, split)

    else:
        raise ValueError(f"split must be 'train', 'test', or 'train_test', got '{split}'")


def _load_split(dataset, split):
    data = dataset.train if split == 'train' else dataset.test
    all_series = []
    for idx, entry in enumerate(data):
        item_id    = entry.get('item_id', entry.get('id', f'series_{idx}'))
        start      = entry['start']
        target     = entry['target']
        date_range = pd.date_range(
            start=start.to_timestamp(),
            periods=len(target),
            freq=start.freq
        )
        all_series.append(pd.DataFrame({
            'unique_id': item_id,
            'ds':        date_range,
            'y':         target
        }))
    return pd.concat(all_series, ignore_index=True)


# def gluonts_to_long_dataframe(dataset_name="m4_yearly", split='train'):
#     """
#     Convert GluonTS dataset to long-format pandas DataFrame
    
#     Parameters:
#     -----------
#     dataset_name : str
#         Name of the dataset (e.g., 'm4_yearly', 'm4_quarterly')
#     split : str
#         'train' or 'test'
    
#     Returns:
#     --------
#     pd.DataFrame with columns: ['unique_id', 'ds', 'y']
#     """
#     from gluonts.dataset.repository.datasets import get_dataset
#     dataset = get_dataset(dataset_name)
#     data = dataset.train if split == 'train' else dataset.test
    
#     all_series = []
#     for idx, entry in enumerate(data):
#         item_id = entry.get('item_id', entry.get('id', f'series_{idx}'))

#         start = entry['start']
#         target = entry['target']

#         date_range = pd.date_range(
#             start=start.to_timestamp(), 
#             periods=len(target), 
#             freq=start.freq
#         )
#         series_df = pd.DataFrame({
#             'unique_id': item_id,
#             'ds': date_range,
#             'y': target
#         })
#         all_series.append(series_df)
    
#     df_long = pd.concat(all_series, ignore_index=True)
    
#     return df_long