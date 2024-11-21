import geopandas as gpd
import multiprocessing as mp
import time
import argparse
import shutil
import os
import datetime

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
        prog = 'python download_chirps.py',
        description = (
            'Script to download CHIRPS data from start_year and end_year. '
            'start_year and end_year both are included in the query.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    DEFAULT_NJOBS = min(mp.cpu_count() - 2, 16)

    # last available files as of 2024-10-09
    DEFAULT_BEFORE_DATE_PRELIM = '2024-10-05'
    DEFAULT_BEFORE_DATE_P05 = '2024-08-31'
    
    VALID_PRODUCTS = [fmcf.chcfetch.Products.CHIRPS.P05, 
                      fmcf.chcfetch.Products.CHIRPS.PRELIM]
    VALID_AGGREGATION = list(rtcm.AGGREGATION_DICT.keys())

    parser.add_argument('start_year', action='store', help=f'Start year for fetching the CHIRPS data. Format: YYYY')
    parser.add_argument('end_year', action='store', help=f'End year for fetching the CHIRPS data. Format: YYYY')
    parser.add_argument('-p', '--product', action='store', default='p05', required=False, help=f'[default = p05] CHIRPS product to be fetched. Options: {VALID_PRODUCTS}.')
    parser.add_argument('-d', '--download_folderpath', action='store', required=False, default=None, help=f"[default = {config.FOLDERPATH_DOWNLOAD_CHC_CHIRPS + 'PRODUCT/'}] Path to the folder where files will be downloaded to.")
    parser.add_argument('-j', '--njobs', action='store', default=DEFAULT_NJOBS, required=False, help=f'[default = {DEFAULT_NJOBS}] Number of cores to use for parallel downloads and computation.')
    parser.add_argument('-b', '--before', metavar='DATE_BEFORE', action='store', default=None, required=False, help=f'[default = {DEFAULT_BEFORE_DATE_PRELIM} for prelim | {DEFAULT_BEFORE_DATE_P05} for p05] Date upto which to query the files for. This is to avoid FTP requests provided files before the given date is already present. Options: [YYYY-MM-DD | today]')

    args = parser.parse_args()

    start_time = time.time()

    start_year = int(args.start_year)
    end_year = int(args.end_year)
    product = str(args.product).lower()
    
    if product not in VALID_PRODUCTS:
        raise ValueError(f'Invalid product. Must be from {VALID_PRODUCTS}.')
    
    if args.download_folderpath is None:
        chirps_download_folderpath = {
            'p05': config.FOLDERPATH_DOWNLOAD_CHC_CHIRPS_P05,
            'prelim': config.FOLDERPATH_DOWNLOAD_CHC_CHIRPS_PRELIM,
        }[product]

    if args.before is None:
        before_date = {
            'p05': DEFAULT_BEFORE_DATE_P05,
            'prelim': DEFAULT_BEFORE_DATE_PRELIM,
        }[product]
        before_date = datetime.datetime.strptime(before_date, '%Y-%m-%d')
    elif str(args.before).lower() == 'today':
        before_date = datetime.datetime.today()
    else:
        before_date = datetime.datetime.strptime(str(args.before), '%Y-%m-%d')
    
    njobs = int(args.njobs)
    if njobs <= 0:
        njobs = mp.cpu_count() - 2

    working_folderpath = config.FOLDERPATH_TEMP

    years = list(range(start_year, end_year + 1))

    print("--- inputs ---")
    print(f"start_year: {start_year}")
    print(f"end_year: {end_year}")
    print(f"product: {product}")
    print(f"download_folderpath: {chirps_download_folderpath}")
    print(f"before_date: {before_date.strftime('%Y-%m-%d')}")
    print(f"njobs: {njobs}")
    
    print("--- run ---")

    catalogue_df = fmcf.fetch_missing_chirps_files(
        years = years,
        product = product,
        chc_chirps_download_folderpath = chirps_download_folderpath,
        njobs = njobs,
        before_date = before_date,
    )
    
    end_time = time.time()

    print(f"--- {round(end_time - start_time, 2)} seconds ---")

