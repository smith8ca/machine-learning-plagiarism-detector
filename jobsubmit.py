import boto3
import os
import pandas as pd
import sagemaker
import sys
from sagemaker.sklearn.estimator import SKLearn
from sklearn.metrics import accuracy_score


# Setup session and role; Create S3 bucket
sagemaker_session = sagemaker.Session()
# role = sagemaker.get_execution_role()
role = sys.argv[1]
bucket = sagemaker_session.default_bucket()

# Upload all data to S3
data_dir = 'models'
prefix = 'plag_data'
input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)


# TESTING: Confirm that data is in S3 bucket
# empty_check = []
# for obj in boto3.resource('s3').Bucket(bucket).objects.all():
#     empty_check.append(obj.key)
#     print(obj.key)

# assert len(empty_check) !=0, 'S3 bucket is empty.'
# print('Test passed!')


# Specify an output path
output_path = 's3://{}/{}'.format(bucket, prefix)

estimator = SKLearn(entry_point = 'train.py',
                    source_dir = 'src',
                    role = role,
                    framework_version="0.23-1",
                    py_version="py3",
                    instance_count = 1,
                    instance_type = 'ml.c4.xlarge',
                    sagemaker_session = sagemaker_session,
                    output_path = output_path,
                    )
    

# Train your estimator on S3 training data
estimator.fit({'train': input_data})


# deploy your model to create a predictor
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.t2.medium')
