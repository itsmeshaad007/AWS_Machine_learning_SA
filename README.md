# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained Resnet50 model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can help in leveraging ML engineering to classify images and solve business problems

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the (Kaggle) dataset available.

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.
Dataset link: https://www.kaggle.com/datasets/salader/dogs-vs-cats
API dataset command: kaggle datasets download -d salader/dogs-vs-cats

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data.

-- dogcat image

### Overview of Project steps
* The jupyter notebook "train_and_deploy.ipynb" walks through the implementation of Image Classification Machine Learning Model to classify between Cats and dogs
* We have used data from Kaggle, link is mentioned below: https://www.kaggle.com/datasets/salader/dogs-vs-cats

* We will be using a pretrained Resnet50 model from pytorch vision library (https://pytorch.org/vision/master/generated/torchvision.models.resnet50.html)

* We will add two Fully connected Neural Network layers on top of the above Resnet50 model

* We will use concept of Transfer learning therefore we will be freezing all the existing Convolutional layers in the pretrained Resnet50 model and only change the gradients for the two fully connected layers

* We perform Hyperparameter tuning, to get the optimal best hyperparameters to be used in our model

* We have added configuration for Profiling and Debugging our training model by adding relevant hooks in Training and Testing (evel) phases

* We will then deploy our Model, for deploying we have created customer inference script. The customer inference script will be overriding a few functions that will be used by our deployed endpoint for making inferences/predictions.

* At the end, we would be testing our model with some test images of dogs, to verify if the model is working as per our expectations.

### Files Used
* hpo.py - This script file contains code that will be used by the hyperparameter tuning jobs to train and test the models with different hyperparameters to find the best hyperparameter
* train_model.py - This script file contains the code that will be used by the training job to train and test the model with the best hyperparameters that we got from hyperparameter tuning
* endpoint_inference.py - This script contains code that is used by the deployed endpoint to perform some preprocessing (transformations) , serialization- deserialization and predictions/inferences and post-processing using the saved model from the training job.
* train_and_deploy.ipynb -- This jupyter notebook contains all the code and steps that we performed in this project and their outputs.

## Hyperparameter Tuning
* The Resnet50 Model with two fully connected Neural network layers are used for the image classification problem. Resnet-50 is 50 layers deep NN and is trained on million images of 1000 categories from the ImageNet Database. 

* The optimizer that we will be using for this model is AdamW ( For more info refer : https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html )

* Hence, the hyperparameters selected for tuning were:
* -- Learning rate - default(x) is 0.001 , so we have selected 0.01x to 100x range for the learing rate
* -- eps - defaut is 1e-08 , which is acceptable in most cases so we have selected a range of 1e-09 to 1e-08
* -- Weight decay - default(x) is 0.01 , so we have selected 0.1x to 10x range for the weight decay
* -- Batch size -- selected only two values [ 64, 128 ]

### HyperParameter Tuning job pic:

### Multiple training jobs triggered by the HyperParameter tuning job pic:

### Best Hyperparameter training job pic: 

### Best hyperparameter Training job logs pic:


## Debugging and Profiling
We had set Debugger hook to record and keep track of the loss criterion metrics of the process in training and testing phases.

The plot of the cross entropy loss is shown below:


We can see in the graph that lines are smoothning beyond steps 50
* How would I go about fixing the more?
   * Adjusting pre-trained model
   * Using additional AWS credits, we could try different configuration of Fully connected layers or we could try adding one more layer in the Neural network
   
## Endpoint Metrics
#### Pic

### Results
Results looks good, as we had utilized GPU while hyperparameter tuning and training of the fine-tuned ResNet50 Model.
ml.g4dn.xlarge instance type for the runing the traiing purposes. However while deploying the model to an endpoint we used the "ml.t2.medium" instance type to save cost and resources.

## Model Deployment
* Model was deployed to a "ml.t2.medium" instance type and we used the "endpoint_inference.py" script to setup and deploy our working endpoint.
* For testing purposes , we will be using some test images that we have stored in the "testImages" folder.
* We will be reading in some test images from the folder and try to send those images as input and invoke our deployed endpoint
* We will be doing this via two approaches
  * Firstly using the Prdictor class object
  * Secondly using the boto3 client

## Deployed active endpoint snapshot:

## Sample output returned from endpoint
