# Inventory Monitoring at Distribution Centers

## Project Setup and Installation

AWS Sagemaker, S3, Jupyter notebook, Python PyTorch and other libraries
were used for this project. Detailed description can be found within the
Jupyter notebook. Within SageMaker Studio, we used ml.t3medium instance
as it low cost and the training is done on more powerful instance like
ml.g4dn.xlarge. Jupyter notebook was used for below cases:

-   Download data and upload it to S3 bucket

-   Data exploration on the data

-   Defining hyperparameters, rules, training containers, debugger, and
    profiler

-   Training the job and hyperparameter tuning and running on multiple
    instances

-   Display profiler report and debugger output on best hyperparameter

-   Deploy the endpoints and make predictions.

Different scripts were also used along with Jupyter Notebook to complete
the above process.

-   hpo.py was used to train the model with default parameters and again
    training it on hyperparameters.

-   train.py was used to train the model on the best hyperparameters and
    then used to obtain the profiler report and debugging output and
    used to run multi-instance training.

-   inference.py was used to create the endpoint to get our predictions.

## Data Acquisition and Exploration

**Datasets and Inputs**

For this project, we will be using the [Amazon Bin Image
Dataset](https://registry.opendata.aws/amazon-bin-imagery/)<sup>1</sup>. This
dataset includes over 500,000 bin JPEG images and corresponding JSON
metadata files describing items in bins in Amazon Fulfillment Centers.
The dataset is available on Amazon S3 and can be imported from there. In
this project we will focusing on bin sizes which can contain up to 5
pieces as including all the data would consume heavy resources and as
per the histogram most of the images have pieces less than 10 so 5 would
be a ideal choice for this exercise. It contains \~10,000 images which
would not consume a lot of resources. Below is the histogram and we can
see up to 5 pieces, 3 pieces bins have the highest quantity. The data
was randomly split into training 60%, validation 20% and testing 20% in
the training script.

![Dist](misc-snaps/dist.png)

## Data Preprocessing

Following preprocessing was on the data for our requirement:
•	Load the image from the input data
•	Resize the image to a fixed size
•	Convert the image to a PyTorch tensor
•	Normalize the pixel values of the image
•	Pass the normalized image through the machine learning model for prediction


![Pre](misc-snaps/prepos.png)

## Modelling

**Pipeline**

For this project, we have used Resnet50 pretrained model as it has
pretty good accuracy and speed compared to other models and is smaller
in size. The following steps were done during the modelling:

-   Data was first trained on default hyperparameters, and its
    performance was measured.

-   Hyperparameter tuning was done to the above model, and it was
    trained again with the best hyperparameters.

-   Endpoints were created and prediction were made by querying the
    images.

-   Performance was evaluated compared to benchmark model.

-   Multi-instance training was performed.

**Evaluation Metrics**

In this project, we will be using accuracy and RMSE to evaluate the model and compare it with the benchmark model
accuracy = correct predictions / total predictions
RMSE = sqrt(sum((predicted_count - true_count)^2) / n)
n = Total number of observations

Cross-Entropy Loss

 ![CEL For](misc-snaps/cel_.png)
 
Where p(x) is the true probability distribution and q(x) is the predicted probability distribution
Loss 

![Loss_for](misc-snaps/loss.png) 

where n is the number of samples in the test dataset, yi is the true value of the ith sample, and yi^ is the predicted value of the ith sample.


**Model Training Metrics**

For the model trained with default hyperparameters of learning rate 0.01
and Batch size 64 we get the below results:

Testing Loss: 96.47792582078414

Testing Accuracy: 18

RMSE: 1.3228756555322954

Training the model with different ranges of hyperparameters as below:

![Hyperparamters](misc-snaps/hpo.png)

The best hyperparameter selected were as below:

![Best_Hyperparamters](misc-snaps/best_hpo.png)

We get the below results:

Testing Loss: 50.02496290206909

Testing Accuracy: 8.07575798034668

RMSE: 1.3228756555322954

**Model Benchmarking**

For this project we have used made by author can be found
[here](https://github.com/silverbottlep/abid_challenge)<sup>2</sup>. The
performance of the benchmark is as below:

Testing Accuracy: 55.67

RMSE: 0.866

We can see our model doesn't perform better than the benchmark model and
further improvements are needed.

**Model Deployment and Predictions**

Once we deploy the model, we test to predict it using one of the images
from the set named 767. We can see it predicts to have 3 pieces in the
bin. It is indexed which is the reason it is showing array(\[2\]) which
falls on the third as marked below.

![Predictions](misc-snaps/preds.png)

## Conclusion

The cross-entropy loss graph can be seen below. It is challenging to
understand the model learning because of various spikes at many
intervals caused by small dataset and outliers. This can be improved by
increasing the dataset size and cleaning the dataset to remove the
outlier to help us achieve a better result.

![Cross Entropy Loss](misc-snaps/cel.png)

**References**


1.  *Amazon bin image dataset*. Amazon Bin Image Dataset - Registry of
    Open Data on AWS. (n.d.).
    https://registry.opendata.aws/amazon-bin-imagery/

2.  Silverbottlep. (n.d.). *Silverbottlep/abid_challenge: Amazon bin
    image dataset challenge*. GitHub.
    https://github.com/silverbottlep/abid_challenge
