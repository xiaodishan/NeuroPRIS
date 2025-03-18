import argparse
import json
import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from main import NeuroPRIS

def get_roc_result(data_names, save_prefix, data_name, test_data_path, output_path, num_points):
    if len(data_names) == 0:
        print("Error: Data names cannot be empty.")
        return

    primary_name = data_names[-1]

    model = NeuroPRIS(num_points=num_points)
    model.model_build()

    model_index_path = glob.glob(f"{save_prefix}/{data_name}*/*.index")
    if not model_index_path:
        print(f"No model index file found in path {save_prefix}/{data_name}*/")
        return

    model_weights_path = model_index_path[0].replace('.index', '')

    model.load(model_weights_path)
    model.model_summary()

    test_dataset = tf.data.Dataset.load(test_data_path)
    y_trues = list(map(lambda a: int(a[1]['outputs']), test_dataset))
    test_dataset = test_dataset.batch(64)

    y_preds = model.predict(test_dataset)
    y_preds = [pred[-1] for pred in y_preds]

    print(f'y_trues: {y_trues[:5]}')
    print(f'y_preds: {y_preds[:5]}')

    y_unique = {}
    for k in y_trues:
        y_unique[k] = y_unique.get(k, 0) + 1
    print(f'y_unique: {y_unique}')

    fpr, tpr, _ = roc_curve(y_trues, y_preds)
    auroc = auc(fpr, tpr)
    print(f'AUROC: {auroc}')

    result = {
        'auroc': auroc,
        'data': list(zip(fpr, tpr))
    }

    json_result = json.dumps(result, indent=4)
    used_model = os.path.basename(model_weights_path)
    json_output_path = os.path.join(output_path, f'{primary_name}.json')
    with open(json_output_path, 'w') as fw:
        fw.write(json_result)

    return json_output_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using the trained model.')
    parser.add_argument('--data_name', type=str, required=True, help='Name of the dataset model used.')
    parser.add_argument('--save_prefix', type=str, default='./model', help='Prefix for model save directories.')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to the test dataset.')
    parser.add_argument('--output_path', type=str, required=True, help='Path for storing prediction results.')
    parser.add_argument('--num_points', type=int, default=1426, help='Number of points for 3D coordinates.')
    args = parser.parse_args()

    json_output_path = get_roc_result([args.data_name], args.save_prefix, args.data_name, args.test_data_path, args.output_path, args.num_points)
    print('Prediction results saved at:', json_output_path)