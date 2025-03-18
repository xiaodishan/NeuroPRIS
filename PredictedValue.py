import argparse
import datetime
import json
import csv
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from main import NeuroPRIS
import os

tf.random.set_seed(1234)

num_points = 1426
cus_learning_rate = 2e-3


def locate_checkpoint_index(model_path):
    """Function to locate the .index file and return the checkpoint prefix."""
    index_files = [f for f in os.listdir(model_path) if f.endswith('.index')]
    if not index_files:
        raise FileNotFoundError("No checkpoint index file found.")
    checkpoint_prefix = index_files[0].replace('.index', '')
    return os.path.join(model_path, checkpoint_prefix)


def get_roc_result(data_names, model_path, predict_data_path, output_path):
    result = {}
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    model = NeuroPRIS(num_points=num_points, learning_rate=cus_learning_rate)
    model.model_build()

    checkpoint_prefix = locate_checkpoint_index(model_path)
    model.load(checkpoint_prefix)

    model.model_summary()

    test_dataset = tf.data.Dataset.load(f'{predict_data_path}').batch(64)

    y_trues = []
    seq_ids = []
    for inputs, outputs, info in test_dataset:
        seq_ids.extend(info['seq_ids'].numpy())
        y_trues.extend(outputs['outputs'].numpy())

    y_preds = model.predict(test_dataset)
    y_preds = list(map(lambda a: a[-1], y_preds))

    fpr, tpr, threshold = roc_curve(y_trues, y_preds)
    auroc = auc(fpr, tpr)
    result['auroc'] = auroc
    result['data'] = list(zip(fpr, tpr))

    y_unique = {}
    for k in y_trues:
        y_unique[k] = y_unique.get(k, 0) + 1

    csv_filename = f'{output_path}/predictedvalue_{data_names[-1]}.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['sequence id', 'predicate value'])
        for seq_id, y_pred in zip(seq_ids, y_preds):
            csv_writer.writerow([seq_id.decode('utf-8') if isinstance(seq_id, bytes) else seq_id, y_pred])

    json_result = json.dumps(result, indent=4)
    used_model = model_path.split('/')[-1]
    json_output_path = f'{output_path}/{used_model}.json'
    with open(json_output_path, 'w') as fw:
        fw.write(json_result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model prediction command')
    parser.add_argument('--modelpath', type=str, required=True, help='Path to the model checkpoint directory')
    parser.add_argument('--predictdatapath', type=str, required=True, help='Path to the prediction data directory')
    parser.add_argument('--outputpath', type=str, required=True, help='Path to save the output files')
    parser.add_argument('--dataname', type=str, nargs='*', default=['10_30HEK293_ELAVL'])

    args = parser.parse_args()

    data_names = list(set(args.dataname))
    model_path = args.modelpath
    predict_data_path = args.predictdatapath
    output_path = args.outputpath

    get_roc_result(data_names, model_path, predict_data_path, output_path)