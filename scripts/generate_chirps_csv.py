import geopandas as gpd
import multiprocessing as mp
import time
import argparse
import shutil
import os
import datetime
import warnings

import sys
sys.path.append('..')

import config
import chcfetch.constants
import fetch_missing_chirps_files as fmcf
import read_tifs_create_met as rtcm


def check_if_any_geom_within_chirps_bounds(
    shapes_gdf:gpd.GeoDataFrame
):
    chirps_bounds_gdf = gpd.read_file(
        chcfetch.constants.CHIRPS_V2_P50_BOUNDS_GEOJSON_FILEPATH
    )

    chirps_bounds_epsg_4326 = chirps_bounds_gdf['geometry'].iloc[0]
    
    any_roi_within_chirps_bounds = shapes_gdf.to_crs(
        chirps_bounds_gdf.crs
    ).within(chirps_bounds_epsg_4326).any()

    return any_roi_within_chirps_bounds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python generate_chirps_csv.py',
        description = (
            'Script to generate a csv for chirps data for a given shapefile, '
            'start_date, and end_date. start_date and end_date both are included '
            'in the query.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    DEFAULT_NJOBS = min(mp.cpu_count() - 2, 16)
    
    VALID_PRODUCTS = [fmcf.chcfetch.Products.CHIRPS.P05, 
                      fmcf.chcfetch.Products.CHIRPS.PRELIM]
    VALID_AGGREGATION = list(rtcm.AGGREGATION_DICT.keys())

    parser.add_argument('roi_filepath', action='store', help='Path to the shapefile.')
    parser.add_argument('start_date', action='store', help=f'Start date for querying the CHIRPS data. Format: YYYY-MM-DD')
    parser.add_argument('end_date', action='store', help=f'End date for querying the CHIRPS data. Format: YYYY-MM-DD')
    parser.add_argument('export_filepath', action='store', help='Filepath where the output csv is to be stored.')
    parser.add_argument('-p', '--product', action='store', default='p05', required=False, help=f'[default = p05] CHIRPS product to be fetched. Options: {VALID_PRODUCTS}.')
    parser.add_argument('-d', '--download_folderpath', action='store', required=False, default=None, help=f"[default = {config.FOLDERPATH_DOWNLOAD_CHC_CHIRPS + 'PRODUCT/'}] Path to the folder where files will be downloaded to.")
    parser.add_argument('-a', '--aggregation', action='store', default='mean', required=False, help=f'[default = mean] Aggregation method to reduce CHIRPS values for a given region to a single value. Options: {VALID_AGGREGATION}.')
    parser.add_argument('-j', '--njobs', action='store', default=DEFAULT_NJOBS, required=False, help=f'[default = {DEFAULT_NJOBS}] Number of cores to use for parallel downloads and computation.')
    parser.add_argument('--ignore-missing-dates', action='store_true', help=f'If there are missing dates for requested date range, this option ignores the error and proceeds, except when there are no files present.')
    parser.add_argument('--warn-missing-dates', action='store_true', help=f'If there are missing dates for requested date range, this option raises a warning and proceeds, except when there are no files present.')
    
    args = parser.parse_args()

    start_time = time.time()

    roi_filepath = args.roi_filepath
    start_date = datetime.datetime.strptime(str(args.start_date), '%Y-%m-%d')
    end_date = datetime.datetime.strptime(str(args.end_date), '%Y-%m-%d')
    export_filepath = args.export_filepath
    product = str(args.product).lower()
    
    if product not in VALID_PRODUCTS:
        raise ValueError(f'Invalid product. Must be from {VALID_PRODUCTS}.')
    
    if args.download_folderpath is None:
        chirps_download_folderpath = {
            'p05': config.FOLDERPATH_DOWNLOAD_CHC_CHIRPS_P05,
            'prelim': config.FOLDERPATH_DOWNLOAD_CHC_CHIRPS_PRELIM,
        }[product]

    aggregation = str(args.aggregation).lower()
    if aggregation not in VALID_AGGREGATION:
        raise ValueError(f'Invalid aggregation. Must be from {VALID_AGGREGATION}.')
    
    njobs = int(args.njobs)
    if njobs <= 0:
        njobs = mp.cpu_count() - 2


    if_missing_dates = 'raise'
    if args.ignore_missing_dates:
        if_missing_dates = 'ignore'
    if args.warn_missing_dates:
        if_missing_dates = 'warn'

    working_folderpath = config.FOLDERPATH_TEMP

    shapes_gdf = gpd.read_file(roi_filepath)

    VAL_COL = f'{aggregation} CHIRPS'

    print("--- inputs ---")
    print(f"roi_filepath: {roi_filepath}")
    print(f"start_date: {start_date.strftime('%Y-%m-%d')}")
    print(f"end_date: {end_date.strftime('%Y-%m-%d')}")
    print(f"export_filepath: {export_filepath}")
    print(f"product: {product}")
    print(f"download_folderpath: {chirps_download_folderpath}")
    print(f"aggregation: {aggregation}")
    print(f"njobs: {njobs}")
    print(f"if_missing_dates: {if_missing_dates}")
    
    print("--- run ---")

    catalogue_df = fmcf.generate_chc_chirps_catalogue_df(
        folderpath = chirps_download_folderpath,
    )

    catalogue_df = catalogue_df[
        (catalogue_df[fmcf.COL_DATE] >= start_date) &
        (catalogue_df[fmcf.COL_DATE] <= end_date)
    ]

    total_days_expected = (end_date - start_date).days
    missing_dates_count = total_days_expected - catalogue_df.shape[0]
    if missing_dates_count > 0:
        if if_missing_dates == 'raise':
            raise ValueError(f'{missing_dates_count} dates missing.')
        if if_missing_dates == 'warn':
            warnings.warn(message = f'{missing_dates_count} dates missing.',
                          category = RuntimeWarning)
    
    catalogue_df[rtcm.COL_METHOD] = rtcm.LoadTIFMethod.READ_AND_CROP

    print('Reading tifs and generating csv')
    updated_catalogue_df = rtcm.read_tifs_get_agg_value(
        shapes_gdf = shapes_gdf,
        catalogue_df = catalogue_df,
        val_col = VAL_COL,
        aggregation = aggregation,
        njobs = njobs,
        working_folderpath = working_folderpath,
    )

    if os.path.exists(working_folderpath):
        shutil.rmtree(working_folderpath)

    os.makedirs(os.path.split(export_filepath)[0], exist_ok=True)
    updated_catalogue_df[[
        fmcf.COL_DATE,
        fmcf.COL_YEAR,
        fmcf.COL_DAY,
        VAL_COL,
    ]].to_csv(export_filepath, index=False)

    end_time = time.time()

    print(f"--- {round(end_time - start_time, 2)} seconds ---")

