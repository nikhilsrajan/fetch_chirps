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


TIF_FILEPATH_COL = 'tif_filepath'
FILETYPE_COL = 'filetype'
DATE_COL = 'date'
TIF_EXT = '.tif'
TIF_GZ_EXT = '.tif.gz'
YEAR_COL = 'year'
DAY_COL = 'day'
IS_CORRUPTED_COL = 'is_corrupted'
TYPE_OF_CORRUPTION_COL = 'type_of_corruption'
MULTIPLIER_COL = 'multiplier'


def geoglam_chirps_filename_parser(filename:str):
    year_day_str = filename.split('_')[1].split('.')[-1]
    year = int(year_day_str[:4])
    day = int(year_day_str[4:])
    return {
        YEAR_COL: int(year),
        DAY_COL: int(day),
    }


def chc_chirps_v2_filename_parser(filename:str):
    date_str = filename.replace('chirps-v2.0.', '').replace('.tif.gz', '')
    datetime_obj = datetime.datetime.strptime(date_str, '%Y.%m.%d')
    year = datetime_obj.year
    day = (datetime_obj - datetime.datetime(year, 1, 1)).days + 1
    return {
        YEAR_COL: year,
        DAY_COL: day,
    }


def create_catalogue_df(
    folderpath:str, 
    filename_parser, 
    ignore_extensions:list[str] = None,
    keep_extensions:list[str]=['.tif'],
    tif_filepath_col:str = TIF_FILEPATH_COL,
):
    data = {TIF_FILEPATH_COL: []}
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
    tif_filepath_col:str = TIF_FILEPATH_COL,
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
        catalogue_df[FILETYPE_COL] = TIF_EXT

        catalogue_df = \
        catalogue_df.sort_values(
            by=[YEAR_COL, DAY_COL]
        ).reset_index(drop=True)

        catalogue_df[DATE_COL] = catalogue_df.apply(
            lambda row: datetime.datetime(year=row[YEAR_COL], month=1, day=1) \
                + datetime.timedelta(days=row[DAY_COL] - 1),
            axis=1
        )

        catalogue_df = \
        catalogue_df[catalogue_df[YEAR_COL].isin(years)]

        catalogue_df[MULTIPLIER_COL] = 1 / 100 # geoprepare multiplies tiff with 100 to convert to integer

    return catalogue_df


def generate_chc_chirps_catalogue_df(
    folderpath:str,
    years:list[int],
    tif_filepath_col:str = TIF_FILEPATH_COL,
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
        catalogue_df[FILETYPE_COL] = TIF_GZ_EXT

        catalogue_df = \
        catalogue_df.sort_values(
            by=[YEAR_COL, DAY_COL]
        ).reset_index(drop=True)

        catalogue_df[DATE_COL] = catalogue_df.apply(
            lambda row: datetime.datetime(year=row[YEAR_COL], month=1, day=1) \
                + datetime.timedelta(days=row[DAY_COL] - 1),
            axis=1
        )

        catalogue_df = \
        catalogue_df[catalogue_df[YEAR_COL].isin(years)]

        catalogue_df[MULTIPLIER_COL] = 1 # from source so no multiplier

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
    tif_filepath_col:str = TIF_FILEPATH_COL,
    is_corrupted_col:str = IS_CORRUPTED_COL,
    type_of_corruption_col:str = TYPE_OF_CORRUPTION_COL,
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
    date_col:str = DATE_COL,
    year_col:str = YEAR_COL,
    day_col:str = DAY_COL,
):
    date = row[date_col]
    year = date.year
    day = (date - datetime.datetime(year, 1, 1)).days + 1
    row[year_col] = year
    row[day_col] = day
    return row


def fetch_missing_chirps_v2p0_p05_files(
    years:list[int],
    geoglam_chirps_data_folderpath:str,
    chc_chirps_v2_0_p05_download_folderpath:str,
    njobs:int = mp.cpu_count() - 2,
    overwrite:bool = False,
    tif_filepath_col:str = TIF_FILEPATH_COL,
):
    print('Creating CHIRPS local catalogue.')
    geoglam_chirps_catalogue_df = generate_geoglam_chirps_catalogue_df(
        folderpath = geoglam_chirps_data_folderpath,
        tif_filepath_col = tif_filepath_col,
        years = years,
    )

    chc_chirps_catalogue_df = generate_chc_chirps_catalogue_df(
        folderpath = chc_chirps_v2_0_p05_download_folderpath,
        tif_filepath_col = tif_filepath_col,
        years = years,
    )

    print('Checking how many files in the local CHIRPS catalogue are corrupted.')
    geoglam_chirps_catalogue_df = add_tif_corruption_cols(
        catalogue_df = geoglam_chirps_catalogue_df,
    )
    n_corrupted = geoglam_chirps_catalogue_df[IS_CORRUPTED_COL].sum()
    print(f'Number of corrupted tifs: {n_corrupted}')

    print(f"Querying CHC for p05 CHIRPS files for years={years}")
    chc_fetch_paths_dfs = []
    for _year in tqdm.tqdm(years):
        _res_df = chcfetch.query_chirps_v2_global_daily(
            product = chcfetch.Products.CHIRPS.P05,
            startdate = datetime.datetime(_year, 1, 1),
            enddate = datetime.datetime(_year, 12, 31),
            show_progress = False,
        )
        chc_fetch_paths_dfs.append(_res_df)
        del _res_df
    chc_fetch_paths_df = pd.concat(chc_fetch_paths_dfs).reset_index(drop=True)

    chc_fetch_paths_df = chc_fetch_paths_df.apply(add_year_day_from_date, axis=1)

    valid_downloads_df = geoglam_chirps_catalogue_df[~geoglam_chirps_catalogue_df[IS_CORRUPTED_COL]]

    valid_downloads_df = pd.concat([
        valid_downloads_df, chc_chirps_catalogue_df
    ]).reset_index(drop=True)

    pending_downloads_df = chc_fetch_paths_df[
        ~chc_fetch_paths_df[DATE_COL].isin(valid_downloads_df[DATE_COL])
    ]

    keep_cols = [DATE_COL, YEAR_COL, DAY_COL, tif_filepath_col, FILETYPE_COL, MULTIPLIER_COL]

    if pending_downloads_df.shape[0] > 0:
        print(f'Number of files that need to be downloaded: {pending_downloads_df.shape[0]}')

        pending_downloads_df = chcfetch.download_files_from_paths_df(
            paths_df = pending_downloads_df,
            download_folderpath = chc_chirps_v2_0_p05_download_folderpath,
            njobs = njobs,
            download_filepath_col = tif_filepath_col,
            overwrite = overwrite,
        )
        pending_downloads_df[FILETYPE_COL] = TIF_GZ_EXT

        merged_catalogue_df = pd.concat([
            pending_downloads_df[keep_cols],
            valid_downloads_df[keep_cols],
        ]).sort_values(by=DATE_COL, ascending=True).reset_index(drop=True)
    else:
        merged_catalogue_df = valid_downloads_df[keep_cols]

    return merged_catalogue_df
