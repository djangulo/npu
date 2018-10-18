import datetime
import os
import re
import sys
import pandas as pd
import numpy as np
from IPython.display import display, HTML
from django.utils.text import slugify

__all__ = [
    'read_data_worksheet',
    'read_worksheet_names',
    'read_columns_to_keep',
    'read_npu_data',
    'rename_columns',
    'display_head',
    'read_product_info',
    'extract_product_info',
    'extend_product_info',
    'extract_asc_info',
    'extend_asc_info',
    'extract_part_codes_info',
    'extend_part_codes_info',
    'reduce_dataframe',
    'append_helper_columns',
]


product_cols = [
    'running_model',
    'hq_svc_product',
    'cis_product_code_index',
    'cis_product_index',
    'cis_product_code_cost',
    'cis_product_cost',
    'product_code',
    'product_name',
    'provision_product_code',
    'provision_product_text',
    'chassis',
    'specification_1',
    'specification_2',
    'specification_3',
    'specification_4',
    'specification_5',
    'specification_6',
    'production_subsidiary_code',
    'basic_model',
]

def read_data_worksheet():
    regex = re.compile(pattern=r'[\w\s\b_]*(raw_data)[\w\s\b_]*',
                           flags=re.IGNORECASE)
    return [i for i in read_worksheet_names() if regex.match(i)][0]


def read_worksheet_names(path=None):
    """Reads worksheet_names.txt file into memory."""
    if path:
        return pd.read_table(path).iloc[:,0].tolist()
    return pd.read_table('data/worksheet_names.txt').iloc[:,0].tolist()

def read_columns_to_keep(path=None):
    """Reads columns_to_keep_clean.txt file into memory."""
    if path:
        return pd.read_table(path).iloc[:, 0].tolist()
    return pd.read_table('data/columns_to_keep_clean.txt').iloc[:,0].tolist()


def read_npu_data(path=None, sheet_name=None, **kwargs):
    """
    Reads the dataframe into the global namespace.

    Keyword arguments:
    path -- Path to read the data from. Default data/data.xlsx
    sheet_name -- Data sheet name. Default read_worksheet_names()[1]
    kwargs -- Additional arguments to pass into pandas.read_excel
    """
    if path:
        if sheet_name:
            return pd.read_excel(path,
                                 **kwargs,
                                 sheet_name=sheet_name)
        return pd.read_excel(path,
                             **kwargs,
                             sheet_name=read_data_worksheet())
    return pd.read_excel('data/data.xlsx',
                         **kwargs,
                         sheet_name=read_data_worksheet())

    
def rename_columns(df):
    """
    Renames the dataframe columns into a slug-like variant 
    replacing dashes ("-") for underscores ("_").

    Keyword arguments:
    df -- Dataframe whose columns will be renamed.
    """
    return df.rename(
        columns=lambda x: slugify(x).replace('-', '_').replace('\t', '')
    )

def display_head(df, lines=5):
    """
    Convenience wrapper around the display(HTML(df.to_html())) combo.

    Keyword arguments:
    df -- DataFrame to display
    lines -- no. of lines to display. Default 5.
    """
    display(HTML(df.head(lines).to_html()))


def read_product_info(path=None):
    """
    Reads product_info.csv file into memory as a pandas dataframe.
    """
    if path:
        return pd.read_csv(path)
    return pd.read_csv('data/product_info.csv')

def extract_product_info(df, save=None, **kwargs):
    """
    Creates a product information dataframe from the df passed.

    Keyword arguments:
    df -- Dataframe from which to extract the product information
    save -- Boolean or string. If passed, will trigger a to_csv call.
            If boolean, it will default on data/product_info.csv, if
            string, it will save the file to that path.
    kwargs -- Additional arguments to pass into pandas.to_csv if the
              save parameter is not None.
    """
    ndf = (
        pd.DataFrame(df.loc[:, 'running_model'].unique())
            .join(pd.DataFrame(df.loc[:,product_cols[1:]]), how='left')
            .rename(columns={0: 'running_model'})
    )
    if save:
        if type(save) == str:
            if not save.endswith('.csv'):
                save += '.csv'
            ndf.to_csv(save, **kwargs)
        else:
            ndf.to_csv('data/product_info.csv', **kwargs)
    return ndf

def extend_product_info(info1, info2, save=None, **kwargs):
    """
    Extends the product info dataframe, while checking for duplicates.

    Keyword arguments:
    info1 -- Product information dataframe to be extended.
    info2 -- Product information dataframe to be appended.
    save -- Boolean or string. If passed, will trigger a to_csv call.
            If boolean, it will default on data/product_info.csv, if
            string, it will save the file to that path.
    kwargs -- Additional arguments to pass into pandas.to_csv if the
              save parameter is not None.
    """
    models1 = info1.loc[:, 'running_model']
    models2 = info2.loc[:, 'running_model']
    new_models = (
        models2[~pd.Series(models2.unique()).isin(list(models1.values))]
    )
    new_models_df = (
        pd.DataFrame(new_models)
        .join(pd.DataFrame(info2.loc[:, product_cols[1:]]), how='left')
    )
    ndf = info1.append(new_models_df)
    if save:
        if type(save) == str:
            if not save.endswith('.csv'):
                save += '.csv'
            ndf.to_csv(save, **kwargs)
        else:
            ndf.to_csv('data/product_info.csv', **kwargs)
    return ndf

def extract_asc_info(df, save=None, **kwargs):
    """
    Creates an Authorized Service Center (ASC) information dataframe
    from the df passed.

    Keyword arguments:
    df -- Dataframe from which to extract the ASC information.
    save -- Boolean or string. If passed, will trigger a to_csv call.
            If boolean, it will default on data/asc_info.csv, if
            string, it will save the file to that path.
    kwargs -- Additional arguments to pass into pandas.to_csv if the
              save parameter is not None.
    """
    ndf = (
        pd.DataFrame(df.loc[:, 'asc_code'].unique())
            .join(pd.DataFrame(df.loc[:,['asc_type', 'asc_name']]), how='left')
            .rename(columns={0: 'asc_code'})
    )
    if save:
        if type(save) == str:
            if not save.endswith('.csv'):
                save += '.csv'
            ndf.to_csv(save, **kwargs)
        else:
            ndf.to_csv('data/asc_info.csv', **kwargs)
    return ndf

def extend_asc_info(asc1, asc2, save=None, **kwargs):
    """
    Extends the Authorized Service Center (ASC) information dataframe,
    while checking for duplicates.

    Keyword arguments:
    asc1 -- ASC information dataframe to be extended.
    asc2 -- ASC information dataframe to be appended.
    save -- Boolean or string. If passed, will trigger a to_csv call.
            If boolean, it will default on data/asc_info.csv, if
            string, it will save the file to that path.
    kwargs -- Additional arguments to pass into pandas.to_csv if the
              save parameter is not None.
    """
    codes1 = asc1.loc[:, 'asc_code']
    codes2 = asc2.loc[:, 'asc_code']
    new_models = (
        codes2[~pd.Series(codes2.unique()).isin(list(codes1.values))]
    )
    new_models_df = (
        pd.DataFrame(new_models)
        .join(pd.DataFrame(asc2.loc[:, ['asc_name', 'asc_type']]), how='left')
    )
    ndf = asc1.append(new_models_df, sort=True)
    if save:
        if type(save) == str:
            if not save.endswith('.csv'):
                save += '.csv'
            ndf.to_csv(save, **kwargs)
        else:
            ndf.to_csv('data/asc_info.csv', **kwargs)
    return ndf


def extract_part_codes_info(df, save=None, **kwargs):
    """
    Creates a product parts information dataframe from the df passed.

    Keyword arguments:
    df -- Dataframe from which to extract the product parts
          information.
    save -- Boolean or string. If passed, will trigger a to_csv call.
            If boolean, it will default on data/part_codes_info.csv, if
            string, it will save the file to that path.
    kwargs -- Additional arguments to pass into pandas.to_csv if the
              save parameter is not None.
    """
    codes = pd.DataFrame(
        columns=['code', 'desc', 'price', 'location', 'date_of_price'])
    for i in range(1, 6):
        codes = codes.append(
            pd.DataFrame(df.loc[:, 'parts_code%s' % i].unique())
            .join(pd.DataFrame(df.loc[:,[
                'parts_code%s' % i,
                'parts_code%s_desc' % i,
                'parts_price%s' % i,
                'location_no%s' %i,
                'bill_confirm_date',
            ]]), how='left')
            .rename(columns={
                0: 'code',
                'parts_code%s_desc' % i: 'desc',
                'parts_price%s' % i: 'price',
                'location_no%s' %i: 'location',
                'bill_confirm_date': 'date_of_price',
            })
            .drop('parts_code%s' % i, axis=1), sort=True
        )
    if save:
        if type(save) == str:
            codes.to_csv(save, **kwargs)
        else:
            codes.to_csv('data/parts_info%(date)s.csv' % {
                'date': datetime.datetime.today().strftime('%Y-%m-%d')
            }, **kwargs)
    return codes

def extend_part_codes_info(parts1, parts2, save=None, **kwargs):
    """
    Extends the product parts information dataframe, while checking
    for duplicates.

    Keyword arguments:
    parts1 -- Product parts information dataframe to be extended.
    parts2 -- Product parts information dataframe to be appended.
    save -- Boolean or string. If passed, will trigger a to_csv call.
            If boolean, it will default on data/product_parts_info.csv,
            if string, it will save the file to that path.
    kwargs -- Additional arguments to pass into pandas.to_csv if the
              save parameter is not None.
    """
    codes1 = parts1.loc[:, 'code']
    codes2 = parts2.loc[:, 'code']
    new_models = (
        codes2[~pd.Series(codes2.unique()).isin(list(codes1.values))]
    )
    new_models_df = (
        pd.DataFrame(new_models)
        .join(pd.DataFrame(parts2.loc[:, [
            'desc',
            'price',
            'location',
            'date_of_price',
        ]]), how='left')
    )
    ndf = parts1.append(new_models_df, sort=True)
    if save:
        if type(save) == str:
            ndf.to_csv(save, **kwargs)
        else:
            ndf.to_csv('data/part_codes_info.csv', **kwargs)
    return ndf

def reduce_dataframe(df, cols=None):
    """
    Reduces the columns to the desired ones. If the cols parameter
    is missing, will call read_columns_to_keep and use those. It will
    remove duplicated columns, keeping only the first occurrence.
    pandas.DataFrame.drop() was avoided due to it removing all
    occurrences of a column name.

    Keyword arguments:
    df -- Dataframe to reduce.
    cols -- Iterable of columns to keep. Defaults to
            helpers.read_columns_to_keep if None.
    """
    dups = []
    ndf = df.copy()
    columns = ndf.columns.tolist()
    for i, v in enumerate(columns):
        if columns.count(v) > 1:
            if not [x for x in dups if x['name'] == v]:
                dups.append({'name': v, 'indexes': []})
            idx = [dups.index(x) for x in dups if x['name'] == v][0]
            dups[idx]['indexes'].append(i)
            dups[idx]['indexes'] = list(set(dups[idx]['indexes']))
    for i in [max(x['indexes']) for x in dups]:
        ndf = pd.concat([ndf.iloc[:, :i], ndf.iloc[:, (i+1):]], axis=1)
    if cols:
        return ndf.loc[:, cols]
    return ndf.loc[:, read_columns_to_keep()]


def append_helper_columns(df):
    """
    Appends additional utility columns to the dataframe.

    Keyword arguments:
    df -- Dataframe to which to append the data.
    """
    ndf = df.copy()
    ndf['4k_tickets'] = 1
    try:
        ndf['npu_count'] = ndf['parts_used'].apply(lambda x: 1 if x == 'N' else 0)
    except AttributeError:
        print('No column named "npu_count", skipping.')
    try:
        ndf['is_ndf'] = ndf['true_npu'].apply(lambda x: 1 if x == 'NDF' else 0)
    except AttributeError:
        print('No column named "true_npu", skipping.')
    return ndf

def read_categorical_columns(path=None):
    """
    Reads categorical_columns.txt file into memory.
    
    Keyword arguments:
    path -- Path to read the data from. Defaults to 
            "data/categorical_columns.txt" if None.
    """
    if path:
        return pd.read_table(path).iloc[:, 0].tolist()
    return pd.read_table('data/categorical_columns.txt').iloc[:,0].tolist()

def read_continuous_columns(path=None):
    """
    Reads continuous_columns.txt file into memory.
    
    Keyword arguments:
    path -- Path to read the data from. Defaults to 
            "data/continuous_columns.txt" if None.
    """
    if path:
        return pd.read_table(path).iloc[:, 0].tolist()
    return pd.read_table('data/continuous_columns.txt').iloc[:,0].tolist()

def read_date_columns(path=None):
    """
    Reads date_columns.txt file into memory.
    
    Keyword arguments:
    path -- Path to read the data from. Defaults to 
            "data/date_columns.txt" if None.
    """
    if path:
        return pd.read_table(path).iloc[:, 0].tolist()
    return pd.read_table('data/date_columns.txt').iloc[:,0].tolist()

def convert_categorical_columns(df, cols=None):
    """
    Converts columns to 'category' type.

    Keyword arguments:
    df -- Dataframe to convert.
    cols -- Iterable of columns to convert. Defaults to
            helpers.read_categorical_columns if None.
    """
    # import pdb; pdb.set_trace()
    ndf = df.copy()
    if cols:
        for col in cols:
            try:
                ndf[col] = ndf[col].astype('category')
            except AttributeError:
                print('No "%s" column in dataframe, skipping' % col)
    for col in read_categorical_columns():
        try:
            ndf[col] = ndf[col].astype('category')
        except AttributeError:
            print('No "%s" column in dataframe, skipping' % col)
    return ndf

def convert_continuous_columns(df, cols=None):
    """
    Converts columns to 'float' type.

    Keyword arguments:
    df -- Dataframe to convert.
    cols -- Iterable of columns to convert. Defaults to
            helpers.read_categorical_columns if None.
    """
    ndf = df.copy()
    if cols:
        for col in cols:
            try:
                ndf[col] = ndf[col].astype(float)
            except AttributeError:
                print('No "%s" column in dataframe, skipping' % col)
    for col in read_continuous_columns():
        try:
            ndf[col] = ndf[col].astype(float)
        except AttributeError:
            print('No "%s" column in dataframe, skipping' % col)
    return ndf


def convert_date_columns(df, cols=None):
    """
    Converts columns to pandas 'date' type.

    Keyword arguments:
    df -- Dataframe to convert.
    cols -- Iterable of columns to convert. Defaults to
            helpers.read_categorical_columns if None.
    """
    ndf = df.copy()
    if cols:
        for col in cols:
            try:
                ndf[col] = df[col].apply(
                    lambda x: pd.to_datetime(x, format=r'%Y%m%d'))
            except AttributeError:
                print('No "%s" column in dataframe, skipping' % col)
    for col in read_date_columns():
        try:
            ndf[col] = df[col].apply(
                    lambda x: pd.to_datetime(x, format=r'%Y%m%d'))
        except AttributeError:
            print('No "%s" column in dataframe, skipping' % col)
    return ndf


if __name__ == '__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))