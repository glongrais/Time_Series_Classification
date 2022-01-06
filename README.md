# Time_Series_Classification

## Principle

The goal of the project is to implement a classifer for time series human activity. The data from wearable sensor are used for human activity recognition of simple activities and postural transitions.

### Table of Content

[Ressources](#Ressources)  
[Data Management](#data-management)  
[Model Building](#model-building)  
[Conclusion](#conclusion)

## Ressources

* The dataset used can be found [here](https://zenodo.org/record/841301#.Ya-NLvHMLRZ)

The project includes 2 jupyter Notebook files:
````
- dataset_creation.ipynb
- main.ipynb
````

The first notebook handles the data visualisation and the datasets creation. The second one compute the machine learning parts. 

## Data Management

### Data Loading

The first step is to load the data from the original dataset. As mention is the assigment requierement, only the data from the right wrist sensor should be used to perfrom the classification. This sensor data is store in the files ```partX/partXdev2.csv```, so only these files need to be loaded. As required, after the data loaded, only the actvities 1 to 5 are kept in the dataset.

### Data Visualisation

A first plot is used to simply see the raw data and their associated label.  
![raw data plot](https://github.com/glongrais/ml-recruitment/blob/glongrais/Figures/raw_data.png)  

Then I checked if the data was properly distrubuted between the classes.

![nb occurences plot](https://github.com/glongrais/ml-recruitment/blob/glongrais/Figures/nbOcc.png)  
![raw data plot](https://github.com/glongrais/ml-recruitment/blob/glongrais/Figures/totalLen.png)  

As we can see some activities were performed more times than some other ones (Activity 1 compared to Actvity 2 or 3). But when we look at the total time spent to perform the activities its more equitable, so no real need to weitgh the classes.  

Lastly, the repartition of the different classes across the candidate is chacked to make sure that the activities are well distributed all over the dataset.  
![Activities distribution](https://github.com/glongrais/ml-recruitment/blob/glongrais/Figures/actDist.png)  

### Dataset Preparation 

After dropping the useless columns (Timestamp and ID) and splitting the dataset (80% for training and 20% for testing), I created windows from the data. More precisly rolling windows to optimise the amount of data available for the training. All the windows overlapping two activities are removed. Each window is associated to one label corresponding to the activity number. A hot-encoding is performed on the label dataset to fit the classifier outputs.
The size of the window is defined according to the length of recording it represents. This [research paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4029702/) defined the best window sizes between 0.25s and 2s. To better find the optimal window size for each model, 4 datasets are created. Each of them having a different window size. As the sensors recorded data at around 52Hz, the size of the windows are defined as following:  
* Dataset 1: Window size = 13 measures (around 0.25s)
* Dataset 2: Window size = 26 measures (around 0.50s)
* Dataset 3: Window size = 52 measures (around 1.00s)
* Dataset 4: Window size = 104 measures (around 2.00s)

## Model Building

### LSTM

I first tried to train with an LSTM as its a model developped for this kind of cases. It can easily find dependencies from sequence.
I performed hyperparameters tuning on the number of units in the LSTM layer (from 32 to 256) and on the dropout value (from 0.2 to 0.5).
With this model the training accuracy can be very high (aboce 0.97) but the validation accuracy stay between 0.60 and 0.65.  
![LSTM Results](https://github.com/glongrais/ml-recruitment/blob/glongrais/Figures/LSTM_result.png) 

### CNN

According to this [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7026300), a special CNN architecture provide good results for human activities recognition. The architecture is defined as the following:  
![CNN Architecture](https://github.com/glongrais/ml-recruitment/blob/glongrais/Figures/cnn_architecture.png)  

![CNN Results](https://github.com/glongrais/ml-recruitment/blob/glongrais/Figures/CNN_result.png) 

### CNN-LSTM

Then I tried with a CNN-LSTM where some features will be first extracted by the CNN, and then the LSTM will learn how to classify from these features. I performed hyperparameters tuning on the number of units in the LSTM layer (from 32 to 256), the dropout value (from 0.2 to 0.5) after the LSTM layer and the number of filters in the Conv layers (32 or 64).
I observed kinda the same results than with the LSTM.

![CNN-LSTM Results](https://github.com/glongrais/ml-recruitment/blob/glongrais/Figures/CNN-LSTM_result.png) 

## Conclusion 

The 3 models probably overfitted regarding the validation accuracy, even after some hyperparameter tuning. A lead would be to manually compute some features from the raw data and feed to models with these features.


