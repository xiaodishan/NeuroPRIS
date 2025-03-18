import argparse
import datetime
import logging

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

import main

matplotlib.use('TkAgg')

custom_objects = {'CustomSchedule': main.CustomSchedule,
                  'OrthogonalRegularizer': main.OrthogonalRegularizer,
                  'Conv_bn': main.Conv_bn,
                  'Dense_bn': main.Dense_bn,
                  'Dense_Layer': main.Dense_Layer,
                  'Tnet': main.Tnet,
                  }

timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def preprocess_sample(inputs, outputs, seq_ids):
    processed_inputs = inputs
    processed_outputs = outputs
    return processed_inputs, processed_outputs

def get_max_sum(scores, seq, window_length=4, top_n=0):
    res_dict = {}
    min_res = -9999
    left = 0
    flag = len(scores) - window_length
    while left <= flag:
        right = left + window_length
        sub_scores = scores[left: right]
        sub_scores_sum = sum(sub_scores)
        if len(res_dict) <= top_n:
            res_dict[sub_scores_sum] = [left, right]
        else:
            if sub_scores_sum > min_res:
                res_dict[sub_scores_sum] = [left, right]
                min_res = min(list(res_dict.keys()))
                res_dict.pop(min_res)

        left += 1

    sub_seqs = []
    if seq:
        for w in res_dict.values():
            sub_seqs.append(seq[w[0]:w[1]])
    return list(res_dict.values()), list(res_dict.keys()), sub_seqs

def get_lstm_result(model_path, dataset_path, data_name, output_path, seq_info_folder, window_length, num_points=1426, cus_learning_rate=0.001, layer_name='biLstm'):

    NeuroPRIS_instance = main.NeuroPRIS(num_points=num_points, learning_rate=cus_learning_rate)
    NeuroPRIS_instance.model_build(return_seq=True)

    # The rest of your existing code remains unchanged
    ckpt = tf.train.Checkpoint(model=NeuroPRIS_instance.myModel)
    latest_ckpt = tf.train.latest_checkpoint(model_path)
    if latest_ckpt:
        ckpt.restore(latest_ckpt).expect_partial()
        logger.info(f'Model loaded from checkpoint: {latest_ckpt}')
    else:
        raise FileNotFoundError("No checkpoint found")

    my_model = tf.keras.Model(inputs=NeuroPRIS_instance.myModel.inputs,
                              outputs=NeuroPRIS_instance.myModel.get_layer(layer_name).output)

    my_model.summary()

    test_dataset = tf.data.Dataset.load(f'{dataset_path}/predict_dataset')
    logger.info(f'{len(test_dataset)=}')

    seq_ids = list(map(lambda a: str(a[2]['seq_ids'].numpy().decode()), test_dataset))
    logger.info(f'{len(seq_ids)=}')

    seq_map = get_seqs_by_id(seq_info_folder, data_name)
    logger.info(f'{len(seq_map)=}')

    seqs = []
    for id in seq_ids:
        seqs.append(seq_map.get(id, ''))
    logger.info(f'{len(seqs)=}')

    test_dataset = test_dataset.map(preprocess_sample)
    test_dataset = test_dataset.batch(64)

    y_preds = my_model.predict(test_dataset)
    logger.info(f'{y_preds.shape=}')

    y_preds_scores = np.sum(y_preds, axis=-1)
    logger.info(f'{y_preds_scores.shape=}')
    y_preds_scores = y_preds_scores.tolist()

    top_n_res = list(
        get_max_sum(y_preds_score, seq, window_length=window_length, top_n=0) for y_preds_score, seq in zip(y_preds_scores, seqs))

    top_n_windows = []
    top_n_values = []
    top_n_sub_seq = []
    for a, b, c in top_n_res:
        top_n_windows.append(a)
        top_n_values.append(b)
        top_n_sub_seq.append(c)

    res_df = pd.DataFrame(
        {'num': range(len(seq_ids)), 'sequence id': seq_ids, 'base scores': y_preds_scores, 'top window': top_n_windows,
         'sequence': seqs, 'motif': top_n_sub_seq})
    res_df.reset_index(drop=True, inplace=True)

    # Remove rows where the 'sequence' column is empty
    res_df = res_df[res_df['sequence'] != '']

    logger.info(res_df.head())

    used_model = model_path.split('/')[-1]
    res_df_save_path = f'{output_path}/motifresult_{data_name}.csv'
    res_df.to_csv(res_df_save_path, index=None)

def get_seqs_by_id(seq_info_folder, data_name):
    positive_seq_info_txt_name = data_name.replace('_dataset', '_positive.txt')
    positive_seq_info_txt_path = f'{seq_info_folder}/{positive_seq_info_txt_name}'

    seqs_map = {}
    with open(positive_seq_info_txt_path, 'r') as f:
        line = f.readline().strip()
        while line:
            if line.startswith('ENST'):
                seq_id = line
                seq = f.readline().strip()
                seqs_map[seq_id] = seq
                second_info = f.readline().strip()
            line = f.readline().strip()
    return seqs_map

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='motif commands')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint directory')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--data_name', type=str, nargs='*', required=True, help='Name of the data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output files')
    parser.add_argument('--seq_info_folder', type=str, required=True, help='Folder containing sequence info')
    parser.add_argument('--window_length', type=int, required=True, help='Window length for get_max_sum function')

    args = parser.parse_args()

    model_path = args.model_path
    dataset_path = args.dataset_path
    data_name = list(set(args.data_name))[0]
    output_path = args.output_path
    seq_info_folder = args.seq_info_folder
    window_length = args.window_length

    get_lstm_result(model_path=model_path, dataset_path=dataset_path, data_name=data_name, output_path=output_path,
                    seq_info_folder=seq_info_folder, window_length=window_length)