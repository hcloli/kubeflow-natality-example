import argparse
import pandas as pd
import numpy as np
import os
import pickle
import json
import time
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.python.lib.io.file_io import FileIO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Tester')
    parser.add_argument('--output_bucket',
                        type=str,
                        default="ai-roadmap-new-experiments-output",
                        help='GCS path of model.')
    parser.add_argument('--input_bucket',
                        type=str,
                        default="ai-roadmap-new-data",
                        help='GCS path to input bucket')
    parser.add_argument('--test_output_path',
                        type=str,
                        help='GCS path to input bucket')
    parser.add_argument('--train_output_path',
                        type=str,
                        help='GCS path to input bucket')
    parser.add_argument('--project_id',
                        type=str,
                        default="ai-roadmap-new",
                        help='GCS path of model.')
    parser.add_argument('--mode',
                        type=str,
                        choices=["cloud", "local"],
                        default="cloud",
                        help='GCS path of model.')
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    os.makedirs("input", exist_ok=True)

    y_score_path = f"{args.test_output_path}/y_score.csv"
    print(f"Downloading y_score results from {y_score_path}")
    with FileIO(y_score_path, 'r') as y_score_file:
        # noinspection PyTypeChecker
        y_score = np.loadtxt(y_score_file, delimiter=",")
    print(y_score)

    label_encoder_path = f"{args.train_output_path}/target_label_encoder.pkl"
    print(f"Downloading label encoder from {label_encoder_path}")
    with FileIO(label_encoder_path, 'rb') as label_encoder_file:
        le = pickle.load(label_encoder_file)

    y_test_path = f"gs://{args.input_bucket}/natality/test/y_test.csv"
    print(f"Downloading y_test from {y_test_path}")
    with FileIO(y_test_path, 'r') as y_test_file:
        y_test = pd.read_csv(y_test_file)['weight_cat']
    y_test_converted = le.transform(y_test)

    fpr, tpr, thresholds = roc_curve(y_test_converted, y_score[:, 1])
    roc_auc = auc(fpr, tpr)
    df_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
    print(df_roc)
    output_dir = f'gs://{args.output_bucket}/natality/output/roc_curve_{time.time()}'
    roc_file = os.path.join(output_dir, 'roc.csv')
    with FileIO(roc_file, 'w') as f:
        # noinspection PyTypeChecker
        df_roc.to_csv(f, columns=['fpr', 'tpr', 'thresholds'], header=False, index=False)

    if args.mode == 'local':
        print("Plotting ROC curve")
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    local_prefix = '/tmp' if args.mode == 'local' else ''
    print(f"Exporting KubeFlow artifacts to {local_prefix}")
    metadata = {
        'outputs': [{
            'type': 'roc',
            'format': 'csv',
            'schema': [
                {'name': 'fpr', 'type': 'NUMBER'},
                {'name': 'tpr', 'type': 'NUMBER'},
                {'name': 'thresholds', 'type': 'NUMBER'},
            ],
            'source': roc_file
        }]
    }
    with FileIO(f'{local_prefix}/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)
    print("Exported metadata:")
    print(json.dumps(metadata, indent=True))

    metrics = {
        'metrics': [{
            'name': 'roc-auc-score',
            'numberValue': roc_auc,
        }]
    }
    with FileIO(f'{local_prefix}/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("KubeFlow metrics exported:")
    print(json.dumps(metrics, indent=True))
