AWS_REGION := 'us-east-1'
CFN_ARTIFACTS_BUCKET := 'plagiarism-detector-sagemaker-smith8ca'
SAGEMAKER_ROLE := 'AmazonSageMaker-ExecutionRole-20210128T172570'

create_bucket:
	# @aws s3api create-bucket --bucket $(CFN_ARTIFACTS_BUCKET) --region $(AWS_REGION) --create-bucket-configuration LocationConstraint=$(AWS_REGION)
	@aws s3api create-bucket --bucket $(CFN_ARTIFACTS_BUCKET) --region $(AWS_REGION)

download:
	@wget https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c4147f9_data/data.zip
	@unzip ./data.zip
	@rm -rf ./data.zip ./__MACOSX

preprocess:
	DATA_DIR=./data/test_info.csv SAVE_DIR=./models/ python3 ./src/preprocess.py

train_cloud:
	MODE=cloud python3 ./jobsubmit.py $(SAGEMAKER_ROLE)

train_local:
	MODE=local python3 ./jobsubmit.py 

deploy_model:
ifdef JOB_NAME
ifdef STACK_NAME
	@aws cloudformation package --template-file ./infra/app/main.template.yaml --output-template-file ./infra/app/output.yaml --s3-bucket $(CFN_ARTIFACTS_BUCKET) --region $(AWS_REGION)
	TRAINING_JOB_NAME=$(JOB_NAME) STACK_NAME=$(STACK_NAME) python3 ./infra/app/deploy.py
else
	$(error "Please provide following arguments: TRAINING_JOB_NAME=String, STACK_NAME=String")
endif
endif

prediction:
ifdef ENDPOINT_NAME
	ENDPOINT_NAME=$(ENDPOINT_NAME) python3 ./get_predictions.py
else
	$(error "Please provide following argument: ENDPOINT_NAME=String")
endif
