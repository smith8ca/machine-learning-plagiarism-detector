import boto3
import os
import pandas as pd
import sagemaker
import sys
from sagemaker.sklearn.estimator import SKLearn
from sklearn.metrics import accuracy_score

sagemaker_role = sys.argv[1]
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()

data_dir = './models'
prefix = 'sagemaker/plagiarism'
test_key = './models/test.csv'
train_key = './models/train.csv'


train_path = sagemaker_session.upload_data(
    train_key, bucket=bucket, key_prefix=prefix)
test_path = sagemaker_session.upload_data(
    test_key, bucket=bucket, key_prefix=prefix)


def local():
    sklearn = SKLearn(
        entry_point='train.py',
        framework_version="0.23-1",
        py_version="py3",
        sagemaker_role=role,
        source_dir='./src/',
        train_instance_count=1,
        train_instance_type='local',
        hyperparameters={
            'max_depth': 5,
                    'n_estimators': 10
        })

    sklearn.fit({'train': 'file://models/train.csv'})
    predictor = sklearn.deploy(initial_instance_count=1, instance_type='local')
    test_data = pd.read_csv('./models/test.csv', header=None, names=None)
    test_y = test_data.iloc[:, 0]
    test_x = test_data.iloc[:, 1:]
    test_y_preds = predictor.predict(test_x)
    accuracy = accuracy_score(test_y, test_y_preds)
    print('The current accuracy score for the prediction', accuracy)


def cloud():
    sklearn = SKLearn(
        entry_point='train.py',
        framework_version="0.23-1",
        instance_count=1,
        instance_type='ml.c4.xlarge',
        py_version="py3",
        sagemaker_role=role,
        sagemaker_session=sagemaker_session,
        source_dir='./src/')

    sklearn.fit({'train': train_path})


if __name__ == '__main__':
    mode = os.environ.get('MODE')
    local() if mode == 'local' else cloud()
