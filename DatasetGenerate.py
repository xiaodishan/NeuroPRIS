import argparse
import glob
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from tensorflow import keras
from Bio import PDB
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

SEQ_MAX_LENGTH = 150
def get_padding_sequence_indices(sequences: List[str], seq_map: Dict[str, int], max_length: int = SEQ_MAX_LENGTH) -> np.ndarray:
    indices = [list(map(lambda x: seq_map.get(x), list(seq))) for seq in sequences]
    padding_seqs = keras.utils.pad_sequences(indices, value=0, padding='post', maxlen=max_length)
    return padding_seqs

def get_coords_for_seq_from_single_pdb(sequence, pdb_path, num_points=1426):
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_path.split('/')[-1].split('.')[0], pdb_path)
        model = structure[0]
        whole_sequence = ''.join(map(lambda a: a.get_resname(), list(model.get_residues())))
        start_index = whole_sequence.index(sequence)
        end_index = start_index + len(sequence)
        residues = list(model.get_residues())[start_index:end_index]
        coords = []
        for residue in residues:
            for atom in residue:
                if 'C' in atom.get_id() or 'c' in atom.get_id():
                    coords.append(atom.get_coord())
        coords_df = pd.DataFrame(coords)
        coords_np_res = np.array(coords_df.sample(n=num_points, replace=True), dtype=np.float32)
    except Exception as e:
        return None, sequence

    return {sequence: coords_np_res}, None

def parallel_get_coords_for_seq(sequences, pdb_paths_list, num_points=1426, max_workers=60):
    seq_coords_np_res_map = {}
    error_sequences = []
    with ThreadPoolExecutor(max_workers) as executor:
        futures = {executor.submit(get_coords_for_seq_from_single_pdb, s, p): p for s, p in zip(sequences, pdb_paths_list)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            coords_np_res, error_sequence = future.result()
            if coords_np_res:
                seq_coords_np_res_map.update(coords_np_res)
            if error_sequence:
                error_sequences.append(error_sequence)
    return error_sequences, seq_coords_np_res_map

def get_pdb_path_base_on_rna_name(rna_name, pdb_folder):
    pdb_paths = glob.glob(f'{pdb_folder}/{rna_name}.pred.pdb')
    return pdb_paths, bool(pdb_paths)

def get_seq_and_sec_shape_and_3d_coords_list(paths, pdb_folder):
    seq_info_map = {}
    sequences = []
    pdb_paths_list = []
    for path in paths:
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line:
                if line.startswith('ENST'):
                    seq_id = line
                    rna_name = line.split('_')[0]
                    sequence = f.readline().strip()
                    secshape_str = f.readline().strip()
                    pdb_paths, flag = get_pdb_path_base_on_rna_name(rna_name, pdb_folder)
                    if not flag:
                        line = f.readline().strip()
                        continue
                    pdb_paths_list.append(pdb_paths[0])
                    sequences.append(sequence)
                    seq_info_map[sequence] = [seq_id, secshape_str]

                line = f.readline().strip()

    error_sequences, seq_coords_np_res_map = parallel_get_coords_for_seq(sequences, pdb_paths_list)

    final_sequences = list(seq_coords_np_res_map.keys())
    coords_3d_list = list(seq_coords_np_res_map.values())
    final_seq_ids = [seq_info_map[seq][0] for seq in final_sequences]
    secshapes = [seq_info_map[seq][1] for seq in final_sequences]

    return final_sequences, secshapes, coords_3d_list, final_seq_ids

def generate_dataset(positive_data_path, negative_data_path, output_dataset_path, pdb_folder):
    RNA_map = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
    secshape_map = {'.': 1, '(': 2, ')': 2}

    pos_sequences, pos_secshapes, pos_3d_coords, pos_seq_ids = get_seq_and_sec_shape_and_3d_coords_list([positive_data_path], pdb_folder)

    neg_sequences, neg_secshapes, neg_3d_coords, neg_seq_ids = get_seq_and_sec_shape_and_3d_coords_list([negative_data_path], pdb_folder)

    pos_sequences = get_padding_sequence_indices(pos_sequences, RNA_map).tolist()
    pos_secshapes = get_padding_sequence_indices(pos_secshapes, secshape_map).tolist()
    neg_sequences = get_padding_sequence_indices(neg_sequences, RNA_map).tolist()
    neg_secshapes = get_padding_sequence_indices(neg_secshapes, secshape_map).tolist()

    pos_labels = [1] * len(pos_sequences)
    neg_labels = [0] * len(neg_sequences)

    sequences_inputs = pos_sequences + neg_sequences
    secshape_inputs = pos_secshapes + neg_secshapes
    coords_3d_inputs = pos_3d_coords + neg_3d_coords
    labels = pos_labels + neg_labels
    seq_ids = pos_seq_ids + neg_seq_ids

    all_dataset = tf.data.Dataset.from_tensor_slices(({"sequences_inputs": sequences_inputs,
                                                       "secshape_inputs": secshape_inputs,
                                                       "coords_3d_inputs": coords_3d_inputs},
                                                       {"outputs": labels}, {"seq_ids": seq_ids}))

    all_dataset = all_dataset.shuffle(len(labels))
    train_count = int(len(all_dataset) * 0.8)
    val_count = int(len(all_dataset) * 0.1)

    train_dataset = all_dataset.take(train_count)
    val_dataset = all_dataset.skip(train_count).take(val_count)
    test_dataset = all_dataset.skip(train_count + val_count)

    train_dataset.save(f'{output_dataset_path}/train_dataset')
    val_dataset.save(f'{output_dataset_path}/val_dataset')
    test_dataset.save(f'{output_dataset_path}/test_dataset')

    predict_dataset = tf.data.Dataset.from_tensor_slices(({"sequences_inputs": sequences_inputs,
                                                           "secshape_inputs": secshape_inputs,
                                                           "coords_3d_inputs": coords_3d_inputs},
                                                           {"outputs": labels}, {"seq_ids": seq_ids}))
    predict_dataset.save(f'{output_dataset_path}/predict_dataset')

    return output_dataset_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate RNA dataset.')
    parser.add_argument('--positive_data_path', type=str, required=True, help='Path to the positive data file.')
    parser.add_argument('--negative_data_path', type=str, required=True, help='Path to the negative data file.')
    parser.add_argument('--output_dataset_path', type=str, required=True, help='Path where datasets will be saved.')
    parser.add_argument('--pdb_folder', type=str, required=True, help='Path to the PDB folder.')
    args = parser.parse_args()

    # Generate the dataset
    dataset_path = generate_dataset(args.positive_data_path, args.negative_data_path, args.output_dataset_path, args.pdb_folder)
    print('Dataset generated at:', dataset_path)