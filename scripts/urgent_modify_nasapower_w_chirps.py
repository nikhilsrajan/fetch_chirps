import os
import argparse
import pandas as pd
import tqdm

import sys
sys.path.append('..')

import config
import rsutils.utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'urgent_modify_nasapower_w_chirps.py',
        description = (
            'Urgent script to modify _nasapower.csv file to replace PRECTOTCORR with CHIRPS values.'
        ),
        epilog = f"--- Send your complaints to {','.join(config.MAINTAINERS)} ---",
    )
    
    parser.add_argument('raion_chirps_folderpath', action='store', help='Folderpath to CHIRPS files.')
    parser.add_argument('vercye_output_folderpath', action='store', help='Folderpath to VeRCYe output root where _nasapower.csv files are present.')

    args = parser.parse_args()

    raion_chirps_folderpath = str(args.raion_chirps_folderpath)
    vercye_output_folderpath = str(args.vercye_output_folderpath)

    if not os.path.exists(raion_chirps_folderpath):
        raise ValueError(f'Raion CHIRPS folder not found: {raion_chirps_folderpath}')
    
    if not os.path.exists(vercye_output_folderpath):
        raise ValueError(f'VeRYCe output folder not found: {vercye_output_folderpath}')
    
    # getting chirps csv filepaths
    chirps_filepaths = rsutils.utils.get_all_files_in_folder(raion_chirps_folderpath)
    raion_chirps_filepath_dict = {
        _filepath.split('/')[-1].split('.')[0] : _filepath 
        for _filepath in chirps_filepaths 
    }

    # getting veryce nasapower filepaths
    nasapower_csv_filepaths = rsutils.utils.get_all_files_in_folder(
        folderpath = vercye_output_folderpath,
        keep_extensions = ['_nasapower.csv']
    )

    if len(nasapower_csv_filepaths) == 0:
        raise ValueError(f'VeRCYe folder path does not contain any _nasapower.csv files')

    # changing _nasapower.csv files
    for _nasapower_filepath in tqdm.tqdm(nasapower_csv_filepaths):
        _nasapower_df = pd.read_csv(_nasapower_filepath, index_col=0)
        raion_name = _nasapower_filepath.split('/')[-2]
        
        chirps_df = pd.read_csv(raion_chirps_filepath_dict[raion_name], index_col='date')

        _nasapower_df = _nasapower_df.rename(columns={'PRECTOTCORR' : 'previous PRECTOTCORR'})
        _nasapower_df['PRECTOTCORR'] = chirps_df.loc[_nasapower_df.index, 'mean CHIRPS']
        _nasapower_df['PRECTOTCORR'] = chirps_df.loc[_nasapower_df.index, 'mean CHIRPS']

        _nasapower_df.to_csv(_nasapower_filepath)

        del _nasapower_filepath, chirps_df
