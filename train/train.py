import argparse
import pandas as pd
import numpy as np
import pickle
import time
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.python.lib.io import file_io

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML Trainer')
    parser.add_argument('--input_bucket',
                        type=str,
                        default="ai-roadmap-new-data",
                        help='GCS path to input bucket')
    parser.add_argument('--data_path',
                        type=str,
                        default='natality/train',
                        help='Path of x_train.csv and y_train.csv within the bucket')
    parser.add_argument('--output_bucket',
                        type=str,
                        default="ai-roadmap-new-experiments-output",
                        help='GCS path to output bucket')
    parser.add_argument('--project_id',
                        type=str,
                        default="ai-roadmap-new",
                        help='GCS path of model.')
    parser.add_argument('--max_depth',
                        type=int,
                        default=2,
                        help='Random Forrest max_depth')
    parser.add_argument('--n_estimators',
                        type=int,
                        default=100,
                        help='Random Forrest number of estimators')
    parser.add_argument('--drop_features',
                        type=str,
                        default="None",
                        help='Comma separated list of features to drop')
    args = parser.parse_args()

    x_train_csv_path = f'gs://{args.input_bucket}/{args.data_path}/x_train.csv'
    y_train_csv_path = f'gs://{args.input_bucket}/{args.data_path}/y_train.csv'

    print(f"Loading train features from {x_train_csv_path}")
    with file_io.FileIO(x_train_csv_path, 'r') as x_train_file:
        X_train = pd.read_csv(x_train_file)
    print(f"Loading train target from {y_train_csv_path}")
    with file_io.FileIO(y_train_csv_path, 'r') as x_train_file:
        y_train = pd.read_csv(x_train_file)['weight_cat']

    if args.drop_features != 'None':
        features_to_drop = args.drop_features.split(",")
        print(f"Removing features {features_to_drop}")
        X_train = X_train.drop(features_to_drop, axis=1)

    le = LabelEncoder()
    le.fit(y_train)
    y = le.transform(y_train)
    print("Classes: %s" % str(le.classes_))
    print("Train: %s." % str(X_train))
    print("Target: %s." % str(y_train))

    print("Training classifier with max_depth %d..." % args.max_depth)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    pipeline = Pipeline([
        ('impute', imp_mean),
        ('clf', RandomForestClassifier(max_depth=args.max_depth,
                                       n_estimators=args.n_estimators,
                                       random_state=0))
    ])
    pipeline.fit(X_train, y)
    print("Classifier trained")
    train_output_path = f'gs://{args.output_bucket}/natality/output/train_{time.time()}'
    model_path = f"{train_output_path}/model.joblib"
    with file_io.FileIO(model_path, 'wb') as model_file:
        dump(pipeline, model_file)
    print(f"Classifier uploaded to {model_path}")

    target_label_output_path = f'{train_output_path}/target_label_encoder.pkl'
    with file_io.FileIO(target_label_output_path, "wb") as label_pickle_file:
        pickle.dump(le, label_pickle_file)
    print(f"Label encoder uploaded to {target_label_output_path}")

    with open("/tmp/train_output_path.txt", "w") as train_output_path_file:
        train_output_path_file.write(train_output_path)
