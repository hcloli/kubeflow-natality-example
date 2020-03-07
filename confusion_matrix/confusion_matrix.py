import argparse
import os
import numpy as np
import pandas as pd
import json
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.python.lib.io.file_io import FileIO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Tester')
    parser.add_argument('--input_bucket',
                        type=str,
                        default="ai-roadmap-new-data",
                        help='GCS path to input bucket')
    parser.add_argument('--output_bucket',
                        type=str,
                        default="ai-roadmap-new-experiments-output",
                        help='GCS path of model.')
    parser.add_argument('--test_output_path',
                        type=str,
                        default="ai-roadmap-new-experiments-output",
                        help='GCS path output bucket.')
    parser.add_argument('--project_id',
                        type=str,
                        default="ai-roadmap-new",
                        help='GCS path of model.')
    parser.add_argument('--mode',
                        type=str,
                        choices=["cloud", "local"],
                        default="cloud",
                        help='GCS path of model.')
    parser.add_argument('--model_input_path',
                        type=str,
                        default="model",
                        help='GCS path of model.')
    args = parser.parse_args()

    y_predict_labels_path = f"{args.test_output_path}/y_predict_labels.csv"
    print(f"Downloading y_predict_labels results from {y_predict_labels_path}")
    with FileIO(y_predict_labels_path, 'r') as y_predict_labels_file:
        # noinspection PyTypeChecker
        y_predict_labels = np.loadtxt(y_predict_labels_file, delimiter=",", dtype=str)
    print(y_predict_labels)

    y_test_path = f"gs://{args.input_bucket}/natality/test/y_test.csv"
    print(f"Downloading y_test from {y_test_path}")
    with FileIO(y_test_path, 'r') as y_test_file:
        y_test = pd.read_csv(y_test_file)['weight_cat']
    print(y_test)

    vocab = ["LOW", "NORMAL"]
    cm = confusion_matrix(y_test, y_predict_labels, labels=vocab)
    print(cm)

    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))

    print(data)

    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    output_dir = f'gs://{args.output_bucket}/natality/output/conf_matrix/roc_curve_{time.time()}/'
    cm_file = os.path.join(output_dir, 'cm.csv')
    with FileIO(cm_file, 'w') as f:
        # noinspection PyTypeChecker
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

    local_prefix = '/tmp' if args.mode == 'local' else ''
    print(f"Exporting KubeFlow artifacts to {local_prefix}")
    metadata = {
        'outputs': [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': cm_file,
            # Convert vocab to string because for boolean values we want "True|False" to match csv data.
            'labels': list(map(str, vocab)),
        }]
    }
    with FileIO(f'{local_prefix}/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)
    print("Exported metadata (8):")
    print(json.dumps(metadata, indent=True))

    accuracy = accuracy_score(y_test, y_predict_labels)
    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue': accuracy,
            'format': "PERCENTAGE",
        }]
    }
    with FileIO(f'{local_prefix}/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Exported metrics:")
    print(json.dumps(metrics, indent=True))
