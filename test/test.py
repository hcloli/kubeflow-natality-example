import argparse
import time
import pandas as pd
import numpy as np
import pickle
from joblib import load
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
                        help='GCS path to output bucket')
    parser.add_argument('--train_output_path',
                        type=str,
                        help='GCS path to trained model')
    parser.add_argument('--mode',
                        type=str,
                        default="local",
                        help='GCS path of model.')
    parser.add_argument('--project_id',
                        type=str,
                        default="ai-roadmap-new",
                        help='GCS path of model.')
    parser.add_argument('--drop_features',
                        type=str,
                        default="None",
                        help='Comma separated list of features to drop')
    args = parser.parse_args()

    x_test_csv_path = f"gs://{args.input_bucket}/natality/test/x_test.csv"
    y_test_csv_path = f"gs://{args.input_bucket}/natality/test/y_test.csv"

    print(f"Loading test features from {x_test_csv_path}")
    with FileIO(x_test_csv_path, 'r') as x_test_file:
        X_test = pd.read_csv(x_test_file)
    print(f"Loading test target from {y_test_csv_path}")
    with FileIO(y_test_csv_path, 'r') as y_test_file:
        y_test = pd.read_csv(y_test_file)

    if args.drop_features != 'None':
        features_to_drop = args.drop_features.split(",")
        print(f"Removing features {features_to_drop}")
        X_test = X_test.drop(features_to_drop, axis=1)

    print("Test data:")
    print(str(X_test))
    print("Test target:")
    print(str(y_test))
    print()

    model_path = f'{args.train_output_path}/model.joblib'
    print(f"Downloading model from {model_path}")
    with FileIO(model_path, 'rb') as model_file:
        loaded_pipeline = load(model_file)
    print("Predicting test")
    y_score = loaded_pipeline.predict_proba(X_test)
    print(y_score)

    y_predict = loaded_pipeline.predict(X_test)
    print(y_predict)

    label_encoder_path = f'{args.train_output_path}/target_label_encoder.pkl'
    print(f"Downloading label encoder from {label_encoder_path}")
    with FileIO(label_encoder_path, 'rb') as label_encoder_file:
        le = pickle.load(label_encoder_file)
    y_predict_labels = le.inverse_transform(y_predict)
    print(y_predict_labels)

    test_output_path = f'gs://{args.output_bucket}/natality/output/test_{time.time()}'

    y_score_path = f'{test_output_path}/y_score.csv'
    print(f"Uploading y_score results to {y_score_path}")
    with FileIO(y_score_path, 'w') as y_score_file:
        np.savetxt(y_score_file, y_score, delimiter=",")

    y_predict_labels_path = f'{test_output_path}/y_predict_labels.csv'
    print(f"Uploading y_predict labels results to {y_predict_labels_path}")
    with FileIO(y_predict_labels_path, 'w') as y_predict_labels_file:
        np.savetxt(y_predict_labels_file, y_predict_labels, delimiter=",", fmt="%s")

    with FileIO("/tmp/test_output_path.txt", "w") as test_output_path_file:
        test_output_path_file.write(test_output_path)

    print("Done!")
