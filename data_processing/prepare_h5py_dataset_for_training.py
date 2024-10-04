'''
This code file is from PromptMR repository (https://github.com/hellopipu/PromptMR)
'''
import os
import sys
import pathlib

sys.path.insert(0, os.path.dirname(os.path.dirname(pathlib.Path(__file__).parent.absolute())))

import shutil
import argparse
import numpy as np
from utils import zf_recon
import h5py
import glob
from os.path import join
from tqdm import tqdm

def split_train_val(h5_folder, train_ratio):
    train_folder = join(h5_folder, 'train')
    val_folder = join(h5_folder, 'val')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    cases = [case for case in os.listdir(h5_folder) if case.startswith('P')]

    np.random.seed(42)  # For reproducibility
    np.random.shuffle(cases)

    train_num = int(len(cases) * train_ratio)

    for i, case in enumerate(cases):
        case_folder = join(h5_folder, case)
        if os.path.exists(case_folder):
            if i < train_num:
                shutil.move(case_folder, train_folder)
            else:
                shutil.move(case_folder, val_folder)

def process_and_save_mat_files(mat_files, save_folder_name):
    for ff in tqdm(mat_files):
        save_path = ff.replace('FullSample', save_folder_name).replace('.mat', '.h5')
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        kdata, image = zf_recon(ff)

        # Open the HDF5 file in write mode
        file = h5py.File(save_path, 'w')

        # Create a dataset
        save_kdata = kdata.reshape(-1, kdata.shape[2], kdata.shape[3], kdata.shape[4]).transpose(0, 1, 3, 2)
        file.create_dataset('kspace', data=save_kdata)

        save_image = image.reshape(-1, image.shape[2], image.shape[3]).transpose(0, 2, 1)
        file.create_dataset('reconstruction_rss', data=save_image)
        file.attrs['max'] = image.max()
        file.attrs['norm'] = np.linalg.norm(image)

        file.attrs['patient_id'] = save_path.split('ChallengeData/')[-1]
        file.attrs['shape'] = kdata.shape
        file.attrs['padding_left'] = 0
        file.attrs['padding_right'] = save_kdata.shape[3]
        file.attrs['encoding_size'] = (save_kdata.shape[2], save_kdata.shape[3], 1)
        file.attrs['recon_size'] = (save_kdata.shape[2], save_kdata.shape[3], 1)

        # Close the file
        file.close()

if __name__ == '__main__':
    argv = sys.argv
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="/media/ruru/dd2c0fe5-b971-4e64-a050-e13627f23931/work/MICCAI/miccai2024/MICCAIChallenge2024/ChallengeData/MultiCoil",
        help="Path to the multi-coil MATLAB folder",
    )

    parser.add_argument(
        "--h5py_folder",
        type=str,
        default="h5_FullSample",
        help="the folder name to save the h5py files",
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of training data split"
    )

    args = parser.parse_args()
    data_path = args.data_path
    save_folder_name = args.h5py_folder

    # aorta_matlab_folder = join(data_path, "Aorta/TrainingSet/FullSample")
    # cine_matlab_folder = join(data_path, "Cine/TrainingSet/FullSample")
    # mapping_matlab_folder = join(data_path, "Mapping/TrainingSet/FullSample")
    # tagging_matlab_folder = join(data_path, "Tagging/TrainingSet/FullSample")
    #
    # assert os.path.exists(aorta_matlab_folder), f"Path {aorta_matlab_folder} does not exist."
    # assert os.path.exists(cine_matlab_folder), f"Path {cine_matlab_folder} does not exist."
    # assert os.path.exists(mapping_matlab_folder), f"Path {mapping_matlab_folder} does not exist."
    # assert os.path.exists(tagging_matlab_folder), f"Path {tagging_matlab_folder} does not exist."

    # folder = ['P070', 'P071', 'P072', 'P073', 'P074', 'P075', 'P076', 'P077', 'P079', 'P081', 'P082', 'P083', 'P084', 'P085', 'P087', 'P088', 'P089', 'P091', 'P093', 'P094', 'P095', 'P112', 'P113', 'P114', 'P116']
    folder = ['P117']
    for case in folder:
        cine_matlab_folder=os.path.join('/media/ruru/dd2c0fe5-b971-4e64-a050-e13627f23931/work/MICCAI/miccai2024/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample', case)
        # 0. get input file list
        # f_aorta = sorted(glob.glob(join(aorta_matlab_folder, '**/*.mat'), recursive=True))
        f = sorted(glob.glob(join(cine_matlab_folder, '**/*.mat'), recursive=True))
        # f_mapping = sorted(glob.glob(join(mapping_matlab_folder, '**/*.mat'), recursive=True))
        # f_tagging = sorted(glob.glob(join(tagging_matlab_folder, '**/*.mat'), recursive=True))

        print('total number of files: ', len(f))
        # print('aorta cases: ', len(os.listdir(aorta_matlab_folder)), ' , aorta files: ', len(f_aorta))
        print('cine cases: ', len(os.listdir(cine_matlab_folder)), ' , cine files: ', len(f))
        # print('mapping cases: ', len(os.listdir(mapping_matlab_folder)), ' , mapping files: ', len(f_mapping))
        # print('tagging cases: ', len(os.listdir(tagging_matlab_folder)), ' , tagging files: ', len(f_tagging))

        # 1. save as fastMRI style h5py files
        process_and_save_mat_files(f, save_folder_name)

        # 2. split into training and validation sets
        # h5_aorta_folder = aorta_matlab_folder.replace('FullSample', save_folder_name)
        h5_cine_folder = cine_matlab_folder.replace('FullSample', save_folder_name)
        # h5_mapping_folder = mapping_matlab_folder.replace('FullSample', save_folder_name)
        # h5_tagging_folder = tagging_matlab_folder.replace('FullSample', save_folder_name)

        # split_train_val(h5_aorta_folder, train_ratio=args.train_ratio)
        # split_train_val(h5_cine_folder, train_ratio=args.train_ratio)
        # split_train_val(h5_mapping_folder, train_ratio=args.train_ratio)
        # split_train_val(h5_tagging_folder, train_ratio=args.train_ratio)
