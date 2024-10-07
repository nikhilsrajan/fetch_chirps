import geopandas as gpd
import multiprocessing as mp
import time
import argparse
import shutil
import os

import sys
sys.path.append('..')

import config
import chcfetch.constants
import fetch_missing_chirps_files as fmcf
import read_tifs_create_met as rtcm


def parse_args():
    roi_shapefile = sys.argv[1]
    start_year = int(sys.argv[2])
    end_year = int(sys.argv[3])
    geoglam_chirps_data_folderpath = sys.argv[4]
    chc_chirps_v2_0_p05_download_folderpath = sys.argv[5]
    reference_tif_filepath = sys.argv[6]
    aggregation = sys.argv[7]
    njobs = int(sys.argv[8])
    export_filepath = sys.argv[9]
    working_folderpath = sys.argv[10]

    return roi_shapefile, start_year, end_year, \
        geoglam_chirps_data_folderpath, \
        chc_chirps_v2_0_p05_download_folderpath, \
        reference_tif_filepath, \
        aggregation, njobs, export_filepath, \
        working_folderpath


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
    """
    args:
    - roi_shapefile
    - start_year
    - end_year
    - geoglam_chirps_folderpath
    - chc_chirps_download_folderpath
    - reference_tif_filepath
    - aggregation
    - njobs
    - export_filepath
    - working_folderpath
    """

    parser = argparse.ArgumentParser(
        prog = 'python generate_chirps_csv.py',
        description = (
            'Script to generate a csv for chirps data for a given shapefile, '
            'start_year, and end_year. start_year and end_year both are included '
            'in the query.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )

    DEFAULT_NJOBS = min(mp.cpu_count() - 2, 16)
    VALID_PRODUCTS = [fmcf.chcfetch.Products.CHIRPS.P05, 
                      fmcf.chcfetch.Products.CHIRPS.PRELIM]
    VALID_AGGREGATION = list(rtcm.AGGREGATION_DICT.keys())

    parser.add_argument('roi_filepath', action='store', help='Path to the shapefile.')
    parser.add_argument('start_year', action='store', help=f'Start year for fetching the CHIRPS data. Format: YYYY')
    parser.add_argument('end_year', action='store', help=f'End year for fetching the CHIRPS data. Format: YYYY')
    parser.add_argument('export_filepath', action='store', help='Filepath where the output csv is to be stored.')
    parser.add_argument('-p', '--product', action='store', default='p05', required=False, help=f'[default = p05] CHIRPS product to be fetched. Options: {VALID_PRODUCTS}.')
    parser.add_argument('-d', '--download_folderpath', action='store', required=False, default=None, help=f"[default = {config.FOLDERPATH_DOWNLOAD_CHC_CHIRPS + 'PRODUCT/'}] Path to the folder where files will be downloaded to.")
    parser.add_argument('-g', '--geoglam-folderpath', required=False, default=None, action='store', help=f"[path/to/geoglam | default] Folderpath where GEOGLAM data is stored. Files present in the GEOGLAM folder will not be re-downloaded.")
    parser.add_argument('-a', '--aggregation', action='store', default='mean', required=False, help=f'[default = mean] Aggregation method to reduce CHIRPS values for a given region to a single value. Options: {VALID_AGGREGATION}.')
    parser.add_argument('-j', '--njobs', action='store', default=DEFAULT_NJOBS, required=False, help=f'[default = {DEFAULT_NJOBS}] Number of cores to use for parallel downloads and computation.')

    args = parser.parse_args()

    start_time = time.time()

    roi_shapefile = args.roi_filepath
    start_year = int(args.start_year)
    end_year = int(args.end_year)
    export_filepath = args.export_filepath
    product = str(args.product).lower() # not used yet
    
    if product not in VALID_PRODUCTS:
        raise ValueError(f'Invalid product. Must be from {VALID_PRODUCTS}.')
    
    if args.download_folderpath is None:
        chirps_download_folderpath = {
            'p05': config.FOLDERPATH_DOWNLOAD_CHC_CHIRPS_P05,
            'prelim': config.FOLDERPATH_DOWNLOAD_CHC_CHIRPS_PRELIM,
        }[product]
    
    geoglam_chirps_folderpath = args.geoglam_folderpath
    if geoglam_chirps_folderpath is not None:
        geoglam_chirps_folderpath = str(geoglam_chirps_folderpath)
        if geoglam_chirps_folderpath.lower() == 'default' and product == 'p05':
            geoglam_chirps_folderpath = config.FOLDERPATH_GEOGLAM_CHIRPS_GLOBAL
        if geoglam_chirps_folderpath.lower() == 'default' and product == 'prelim':
            print('Warning: No default GEOGLAM folderpath set for product=prelim. Reverting to None.')
            geoglam_chirps_folderpath = None

    aggregation = str(args.aggregation).lower()
    if aggregation not in VALID_AGGREGATION:
        raise ValueError(f'Invalid aggregation. Must be from {VALID_AGGREGATION}.')
    
    njobs = int(args.njobs)
    if njobs <= 0:
        njobs = mp.cpu_count() - 2

    working_folderpath = config.FOLDERPATH_TEMP

    shapes_gdf = gpd.read_file(roi_shapefile)
    years = list(range(start_year, end_year + 1))

    VAL_COL = f'{aggregation} CHIRPS'

    catalogue_df = fmcf.fetch_missing_chirps_files(
        years = years,
        product = product,
        geoglam_chirps_data_folderpath = geoglam_chirps_folderpath,
        chc_chirps_download_folderpath = chirps_download_folderpath,
        njobs = njobs,
    )
    
    reference_tif_filepath = None
    geoglam_catalogue_df = catalogue_df[catalogue_df[fmcf.COL_SOURCE] == fmcf.SOURCE_GEOGLAM]

    if geoglam_catalogue_df.shape[0] > 0:
        reference_tif_filepath = geoglam_catalogue_df[fmcf.COL_TIF_FILEPATH].tolist()[0]

    catalogue_df[rtcm.COL_METHOD] = rtcm.LoadTIFMethod.READ_AND_CROP
    if reference_tif_filepath is not None:
        catalogue_df.loc[
            catalogue_df[fmcf.COL_FILETYPE] == fmcf.EXT_TIF_GZ,
            rtcm.COL_METHOD
        ] = rtcm.LoadTIFMethod.COREGISTER_AND_CROP


    print('Reading tifs and generating csv')
    updated_catalogue_df = rtcm.read_tifs_get_agg_value(
        shapes_gdf = shapes_gdf,
        catalogue_df = catalogue_df,
        val_col = VAL_COL,
        reference_tif_filepath = reference_tif_filepath,
        aggregation = aggregation,
        njobs = njobs,
        working_folderpath = working_folderpath,
    )

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

