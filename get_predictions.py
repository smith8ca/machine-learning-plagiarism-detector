import os
import io

import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME')
CONTENT_TYPE = 'application/python-pickle'
DATA_DIR = 'models'


# Read in test data
test_data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), header=None, names=None)
test_y = test_data.iloc[:,0]
test_x = test_data.iloc[:,1:]

def np2csv(arr):
    csv = io.BytesIO()
    np.savetxt(csv, arr, delimiter=',', fmt='%g')
    return csv.getvalue().decode().rstrip()


payload = np2csv(test_x)

client = boto3.client('sagemaker-runtime')
response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType='text/csv',
    Body=payload
)


test_y_preds = json.loads(response['Body'].read().decode('utf-8'))
print('Accuracy Score: ', accuracy_score(test_y, test_y_preds))
