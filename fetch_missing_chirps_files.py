import os
import pandas as pd
import datetime
import tqdm
import affine
import rasterio
import multiprocessing as mp

import chcfetch.chcfetch as chcfetch
import rsutils.utils as utils


# https://stackoverflow.com/questions/18603270/progress-indicator-during-pandas-operations
tqdm.tqdm.pandas()


COL_TIF_FILEPATH = 'tif_filepath'
COL_FILETYPE = 'filetype'
COL_DATE = 'date'
COL_YEAR = 'year'
COL_DAY = 'day'
COL_IS_CORRUPTED = 'is_corrupted'
COL_TYPE_OF_CORRUPTION = 'type_of_corruption'
COL_MULTIPLIER = 'multiplier'
COL_SOURCE = 'source'

EXT_TIF = '.tif'
EXT_TIF_GZ = '.tif.gz'

SOURCE_CHC = 'chc'
SOURCE_GEOGLAM = 'geoglam'


def geoglam_chirps_filename_parser(filename:str):
    year_day_str = filename.split('_')[1].split('.')[-1]
    year = int(year_day_str[:4])
    day = int(year_day_str[4:])
    return {
        COL_YEAR: int(year),
        COL_DAY: int(day),
    }


def chc_chirps_v2_filename_parser(filename:str):
    date_str = filename.replace('chirps-v2.0.', '').replace('.tif.gz', '')
    datetime_obj = datetime.datetime.strptime(date_str, '%Y.%m.%d')
    year = datetime_obj.year
    day = (datetime_obj - datetime.datetime(year, 1, 1)).days + 1
    return {
        COL_YEAR: year,
        COL_DAY: day,
    }


def create_catalogue_df(
    folderpath:str, 
    filename_parser, 
    ignore_extensions:list[str] = None,
    keep_extensions:list[str]=['.tif'],
    tif_filepath_col:str = COL_TIF_FILEPATH,
):
    data = {COL_TIF_FILEPATH: []}
    for filepath in utils.get_all_files_in_folder(
        folderpath=folderpath,
        ignore_extensions=ignore_extensions,
        keep_extensions=keep_extensions,
    ):
        data[tif_filepath_col].append(filepath)
        filename = os.path.split(filepath)[1]
        parsed = filename_parser(filename=filename)
        for key, value in parsed.items():
            if key not in data.keys():
                data[key] = []
            data[key].append(value)
    catalogue_df = pd.DataFrame(data=data)
    return catalogue_df


def generate_geoglam_chirps_catalogue_df(
    folderpath:str,
    years:list[int],
    tif_filepath_col:str = COL_TIF_FILEPATH,
):
    catalogue_df = create_catalogue_df(
        folderpath = folderpath,
        filename_parser = geoglam_chirps_filename_parser,
        keep_extensions = ['.tif'],
        tif_filepath_col = tif_filepath_col,
    )

    if catalogue_df.shape[0] == 0:
        catalogue_df = pd.DataFrame()
    else:
        catalogue_df[COL_FILETYPE] = EXT_TIF

        catalogue_df = \
        catalogue_df.sort_values(
            by=[COL_YEAR, COL_DAY]
        ).reset_index(drop=True)

        catalogue_df[COL_DATE] = catalogue_df.apply(
            lambda row: datetime.datetime(year=row[COL_YEAR], month=1, day=1) \
                + datetime.timedelta(days=row[COL_DAY] - 1),
            axis=1
        )

        catalogue_df = \
        catalogue_df[catalogue_df[COL_YEAR].isin(years)]

        # geoprepare multiplies tiff with 100 to convert to integer
        catalogue_df[COL_MULTIPLIER] = 1 / 100
        catalogue_df[COL_SOURCE] = SOURCE_GEOGLAM

    return catalogue_df


def generate_chc_chirps_catalogue_df(
    folderpath:str,
    years:list[int],
    tif_filepath_col:str = COL_TIF_FILEPATH,
):
    catalogue_df = create_catalogue_df(
        folderpath = folderpath,
        filename_parser = chc_chirps_v2_filename_parser,
        keep_extensions = ['.tif.gz'],
        tif_filepath_col = tif_filepath_col,
    )

    if catalogue_df.shape[0] == 0:
        catalogue_df = pd.DataFrame()
    else:
        catalogue_df[COL_FILETYPE] = EXT_TIF_GZ

        catalogue_df = \
        catalogue_df.sort_values(
            by=[COL_YEAR, COL_DAY]
        ).reset_index(drop=True)

        catalogue_df[COL_DATE] = catalogue_df.apply(
            lambda row: datetime.datetime(year=row[COL_YEAR], month=1, day=1) \
                + datetime.timedelta(days=row[COL_DAY] - 1),
            axis=1
        )

        catalogue_df = \
        catalogue_df[catalogue_df[COL_YEAR].isin(years)]

        catalogue_df[COL_MULTIPLIER] = 1 # from source so no multiplier
        catalogue_df[COL_SOURCE] = SOURCE_CHC

    return catalogue_df


def check_if_corrupted(tif_filepath):
    is_corrupted = False
    type_of_corruption = None

    CORRUPTED_UNOPENABLE = 'UNOPENABLE'
    CORRUPTED_INVALID_TRANSFORM = 'INVALID_TRANSFORM'

    INVALID_TRANSFORM = affine.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    try:
        with rasterio.open(tif_filepath) as src:
            out_meta = src.meta.copy()
        if out_meta['transform'] == INVALID_TRANSFORM:
            is_corrupted = True
            type_of_corruption = CORRUPTED_INVALID_TRANSFORM
    except rasterio.RasterioIOError:
        is_corrupted = True
        type_of_corruption = CORRUPTED_UNOPENABLE
    
    return is_corrupted, type_of_corruption


def add_tif_corruption_cols(
    catalogue_df:pd.DataFrame, 
    tif_filepath_col:str = COL_TIF_FILEPATH,
    is_corrupted_col:str = COL_IS_CORRUPTED,
    type_of_corruption_col:str = COL_TYPE_OF_CORRUPTION,
    njobs:int=mp.cpu_count() - 2,
):
    if catalogue_df.shape[0] == 0:
        catalogue_df[is_corrupted_col] = []
        catalogue_df[type_of_corruption_col] = []
        return catalogue_df

    with mp.Pool(njobs) as p:
        list_corrupt_stats = list(tqdm.tqdm(
            p.imap(check_if_corrupted, catalogue_df[tif_filepath_col]), 
            total=catalogue_df.shape[0])
        )

    is_corrupted_series, type_of_corruption_series = zip(*list_corrupt_stats)
    catalogue_df[is_corrupted_col] = is_corrupted_series
    catalogue_df[type_of_corruption_col] = type_of_corruption_series
    return catalogue_df


def add_year_day_from_date(
    row, 
    date_col:str = COL_DATE,
    year_col:str = COL_YEAR,
    day_col:str = COL_DAY,
):
    date = row[date_col]
    year = date.year
    day = (date - datetime.datetime(year, 1, 1)).days + 1
    row[year_col] = year
    row[day_col] = day
    return row


# WARNING: This function is doing too many things
def fetch_missing_chirps_files(
    years:list[int],
    product:str,
    chc_chirps_download_folderpath:str,
    geoglam_chirps_data_folderpath:str = None,
    njobs:int = mp.cpu_count() - 2,
    overwrite:bool = False,
    tif_filepath_col:str = COL_TIF_FILEPATH,
):
    VALID_PRODUCTS = [chcfetch.Products.CHIRPS.P05, chcfetch.Products.CHIRPS.PRELIM]
    if product not in VALID_PRODUCTS:
        raise ValueError(f'Invalid product. Must be from {VALID_PRODUCTS}')

    print('Creating CHIRPS local catalogue.')

    chc_chirps_catalogue_df = generate_chc_chirps_catalogue_df(
        folderpath = chc_chirps_download_folderpath,
        tif_filepath_col = tif_filepath_col,
        years = years,
    )

    geoglam_folderpath_provided = geoglam_chirps_data_folderpath is not None

    if geoglam_folderpath_provided:
        geoglam_chirps_catalogue_df = generate_geoglam_chirps_catalogue_df(
            folderpath = geoglam_chirps_data_folderpath,
            tif_filepath_col = tif_filepath_col,
            years = years,
        )
        print('Checking how many files in the local CHIRPS catalogue are corrupted.')
        geoglam_chirps_catalogue_df = add_tif_corruption_cols(
            catalogue_df = geoglam_chirps_catalogue_df,
        )
        n_corrupted = geoglam_chirps_catalogue_df[COL_IS_CORRUPTED].sum()
        print(f'Number of corrupted tifs: {n_corrupted}')
    else:
        raise NotImplementedError('Yet to implement the case when geoglam_chirps_data_folderpath is None.')

    print(f"Querying CHC for {product} CHIRPS files for years={years}")
    chc_fetch_paths_dfs = []
    for _year in tqdm.tqdm(years):
        _res_df = chcfetch.query_chirps_v2_global_daily(
            product = product,
            startdate = datetime.datetime(_year, 1, 1),
            enddate = datetime.datetime(_year, 12, 31),
            show_progress = False,
        )
        chc_fetch_paths_dfs.append(_res_df)
        del _res_df
    chc_fetch_paths_df = pd.concat(chc_fetch_paths_dfs).reset_index(drop=True)

    chc_fetch_paths_df = chc_fetch_paths_df.apply(add_year_day_from_date, axis=1)

    if geoglam_folderpath_provided:
        valid_downloads_df = geoglam_chirps_catalogue_df[~geoglam_chirps_catalogue_df[COL_IS_CORRUPTED]]

        valid_downloads_df = pd.concat([
            valid_downloads_df, chc_chirps_catalogue_df
        ]).reset_index(drop=True)

        pending_downloads_df = chc_fetch_paths_df[
            ~chc_fetch_paths_df[COL_DATE].isin(valid_downloads_df[COL_DATE])
        ]
    else:
        pending_downloads_df = chc_fetch_paths_df

    keep_cols = [COL_DATE, COL_YEAR, COL_DAY, tif_filepath_col, COL_FILETYPE, COL_MULTIPLIER]

    if pending_downloads_df.shape[0] > 0:
        print(f'Number of files that need to be downloaded: {pending_downloads_df.shape[0]}')

        pending_downloads_df = chcfetch.download_files_from_paths_df(
            paths_df = pending_downloads_df,
            download_folderpath = chc_chirps_download_folderpath,
            njobs = njobs,
            download_filepath_col = tif_filepath_col,
            overwrite = overwrite,
        )
        pending_downloads_df[COL_FILETYPE] = EXT_TIF_GZ

        merged_catalogue_df = pd.concat([
            pending_downloads_df[keep_cols],
            valid_downloads_df[keep_cols],
        ]).sort_values(by=COL_DATE, ascending=True).reset_index(drop=True)
    else:
        merged_catalogue_df = valid_downloads_df[keep_cols]

    return merged_catalogue_df
