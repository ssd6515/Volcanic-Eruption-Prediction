# Volcanic Eruption Prediction

* This repository holds an attempt to predict the time for the next volcanic eruption for each volcano from Kaggle challenge. 
* link - https://www.kaggle.com/competitions/predict-volcanic-eruptions-ingv-oe/overview/description

## Overview


 * **Definition of the tasks / challenge:**  The task is to predict when a volcano's next eruption will occur by analyzing a large geophysical dataset collected by sensors deployed on active volcanoes. 
 
 * **Approach:** The approach in this repository formulates the problem as a regression task, using deep neural networks as the model with the time series of seismic activity as input.  
 
 * **Summary of the performance achieved:** Our best model was able to predict the time for volcano's next eruption with the metric(MAE) of 11632481.80263158.

## Summary of Workdone

### Data

* Data:
  * Files:
   * File descriptions:
       * train.csv - Metadata for the train files.
       
       * [train|test]/*.csv - The data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano.
       
       * sample_submission.csv - a sample submission file in the correct format.
       
      
  * Type:
    * Input: CSV file of following features:
      * segment_id: ID code for the data segment. Matches the name of the associated data file.
      * sensor_1-sensor_10: Each dataframe in train and test file contain 10 sensor data for the volcano. 
      
    * Output:
      * time_to_eruption: The target value, the time until the next eruption.
    
  * Size: 31.25 GB
  
  * Instances (Train, Test, Validation Split): data points: 1485. 605 volcano id for training, 728 for testing, 152 for validation

### Preprocessing / Clean up

Figure 1:

<img width="743" alt="g4" src="https://user-images.githubusercontent.com/89792366/225078119-83325f0a-b970-4013-ad4b-05b68478c6dd.png">

Figure 1: There are a lot of senosrs with missing data with sensor 2 being the highest. Thus, NAN values has been taken care of in preprocess for train_data and test_data for better modeling.

### Feature Engineering

* Added the following features: 
     * sensor_1_mad-sensor_10_mad: The mean absolute deviation values for 10 sensors in each csv file.                      
     * sensor_1_skew-sensor_10_skew: The skewness values for 10 sensors in each csv file.                                  
     * sensor_1_kurt-sensor_10_kurt: The kurtosis values for 10 sensors in each csv file.            
     * sensor_1_nunique-sensor_10_nunique: The unique values for 10 sensors in each csv file.   
     * sensor_1_quantile_05-sensor_10_quantile_05: The 5% quantile values for 10 sensors in each csv file. Similary other quantiles (10,30,70,90,95) were added.
     * sensor_1_fft_power_mean-sensor_10_fft_power_mean: The fast fourier mean values for 10 sensors in each csv file. Similarly, fft feature was used for mean, standard deviation, minimum, maximum, sume of low, middle, high values, mean absolute deviation, skewness, kurtosis, unique, quantile values of 10 sensors in each file.  
     * sensor_1_roll_mean_min-sensor_10_roll_mean_min: The rolling minimum of mean values for 10 sensors in each csv file. Similarly, rolling feature was used for maximum of mean, minimum value in difference between max and min of rolling mean, etc. 
     * sensor_1_first_005-sensor_10_first_005: The minimum values in first 5% of time data for 10 sensors in each csv file. Similarly, the minimum and maximum values for different time ranges (10%,30%,70%,90%,95%) were extracted.
     
    * A total of 570 features were extracted from the dataset.
     
### Data Visualization

Figure 2:

<img width="260" alt="sensor1" src="https://user-images.githubusercontent.com/89792366/225078688-e6702f23-a7c6-412e-ac43-d9af93cfda3f.png">

Figure 2: All of the sensors respond to the volcano showing different variations in each of them. Majority of them show large variation little over 30000.



Figure 3:

<img width="260" alt="sensor2" src="https://user-images.githubusercontent.com/89792366/225078748-1143ca7a-b442-445e-a8c1-0a6d5b121026.png">

Figure 3: Sensor 2 and 8 don't respond or have no data for the volcano. Majority of the other sensors show large variation around 40000.

Thus, looking at both plots an attempt has been made to make a better model by extracting features like the minimum, maximum, median, mean, and standard deviation of each sensor. 



Figure 4:

<img width="463" alt="g3" src="https://user-images.githubusercontent.com/89792366/225078813-e9088fca-2896-44f2-84c8-533eae872d5a.png">

Figure 4: Overall the distribution of the time_to_eruption looks uniform. With exception of few volcanoes taking a longer period of time at the end of the distribution. Thus, the model in my opinion should predict uniform results for the test_data as well.

Figure 5:

<img width="490" alt="dd" src="https://user-images.githubusercontent.com/89792366/234763644-a8b3016b-47f5-45ee-aaa0-04b8eac1f0b5.png">

Figure 5: There are some features that have skewness after feature extraction. Above figure is one example of a skewed feature. These features are scaled using RobustScaler. 
 
### Problem Formulation

* Data:
  * Input: The input is the dataset with added features explained in Preprocessing/Cleanup
  * Output: time_to_eruption
  
  * Models
  
    * XGBRegressor: The XGBRegressor generally classifies the order of importance of each feature used for the prediction. A benefit of using gradient boosting is that after the boosted trees are constructed, it is relatively straightforward to retrieve importance scores for each attribute. Thus, I used this model for understanding the importance of sensors.
    
    * Random Forest Regressor: A number of decision trees are used on distinct subsets of the same dataset, and the average is used to improve the dataset's projected accuracy. The random forest collects the data of each tree and forecasts the future based on the majority of predictions, rather than relying on a single decision tree. Thus, I used this model for branching sensors.

### Training
* Software used:
   * Python packages: numpy, pandas, math, sklearn, seaborn, matplotlib.pyplot, xgboost, joblib
   
* XGB Model:

  The model was created as follows: 
  
  
   <img width="734" alt="model" src="https://user-images.githubusercontent.com/89792366/225089492-1655955b-a3c8-43b0-a911-8f6478b9c2d1.png">
 
  The model was trained with fit method: 
   
  
   <img width="733" alt="train" src="https://user-images.githubusercontent.com/89792366/225089543-57545f4f-2928-4c27-a804-b8391e700731.png">



  The feature importance plot was plotted with their f-score:  
  
  
   <img width="611" alt="dd1" src="https://user-images.githubusercontent.com/89792366/234764466-3d48d10f-4542-4196-94d6-924f4f7b27cf.png"> 
  
  
  This plot shows some of the features with their f-scores with sensor_9_skew having the highest f-score and sensor_1_quantile_10 having the lowest f-score.


* RandomForest Model:


  The model was created as follows:
  
  
   <img width="731" alt="m1" src="https://user-images.githubusercontent.com/89792366/225090629-96adc108-16da-4524-a390-5e5c141f608c.png">
  
  
  The model was trained with fit method:
  
  
   <img width="730" alt="t1" src="https://user-images.githubusercontent.com/89792366/225090682-80d8f99c-33fb-4d1a-a8ad-264dab85dfcf.png">



### Performance Comparison


* The performance metric is mean absolute error(MAE).

* Table:


   <img width="125" alt="ds" src="https://user-images.githubusercontent.com/89792366/234765147-ded67e9f-2053-4e0a-909f-3a6b06615da5.png">

The  following plots show the difference between the difference between true Y_valid and predicted Y_pred from X_valid data.

* XGB plot:


  <img width="822" alt="x" src="https://user-images.githubusercontent.com/89792366/234765427-5bb33900-88e3-4ee7-8e1c-dd693e9b6cf1.png">
  
  
  
* RandomForest plot:


  <img width="827" alt="r" src="https://user-images.githubusercontent.com/89792366/234765458-045f3ed3-f87d-419b-b724-273e7bd79d78.png">
  


### Conclusions

*  From the plots it is seen that the RandomForest model was the best among the 2 models as the predicted data was closer to the true value and condensed than the other models. But still, there is some variation from the true value of validation data in RandomForest model. Overall, the RandomForest model was effective only to the small samples of data used due to storage issues. This may have lead to larger MAE value than expected between 4000000-5000000.


### Future Work

* In future, I can make the models more effective in terms of getting accurate predictions. This can be done by providing a better environment like I was unable to use Sagemaker, which would have resulted in better prediction. Also, I would use an environment where I could potentially incorporate all the available data to create a model. As I had storage issues, I had to use limited data, which might have resulted in large MAE value than expected.

### How to reproduce results


* To reproduce the results:

  * import XGBRegressor, RandomForest.
   
    * The followings commands can be used to import:
      * from xgboost import XGBRegressor
      * from xgboost import plot_importance
      * from sklearn.ensemble import RandomForestRegressor
     
   * Create the train, valid, test dataset as described:
   
   
   <img width="731" alt="t2" src="https://user-images.githubusercontent.com/89792366/225097326-dbc6e58a-0c27-4563-be79-bb9c8a9cbacb.png">
   

   * Create model as described in Training Section.
   
   * Train the model as described in Training Section.
   
   * The predictions can be made as follows for valid and test data to get the feature importance and train/valid loss plot and RMSE.
   
    * Feature Importance Plot:
    
      <img width="733" alt="i" src="https://user-images.githubusercontent.com/89792366/225097680-c3b933d7-26fa-4a32-bbf6-d60c45607dc4.png">
    
    
    * Finally, you can get the plots that show the difference between the difference between true Y_valid and predicted Y_pred from X_valid data as follows:
    
    
     
   
    <img width="724" alt="f" src="https://user-images.githubusercontent.com/89792366/225097907-4569dced-1add-476a-ae9a-64aa77d1f127.png">



    * Repeat this method for other models.
    
    
### Overview of files in repository

  * preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
  * visualization.ipynb: Creates various visualizations of the data.
  * XGBoost.ipynb: Trains the first model and saves model during training.
  * RandomForest.ipynb: Trains the second model and saves model during training.
  * performance.ipynb: loads multiple trained models and compares results.
  * inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.
  * submission_rf.csv - file contaning time_to_eruption for the test data for randomforest model.
  * submission_xgb.csv - file contaning time_to_eruption for the test data for xgbregressor model.

### Software Setup

* Python packages: numpy, pandas, math, sklearn, seaborn, matplotlib.pyplot, xgboost, joblib
* Download seaborn in jupyter - pip install seaborn
* Download xgboost in jupyter - pip install xgboost

### Data

* Download data files required for the project from the following link:
  https://www.kaggle.com/competitions/predict-volcanic-eruptions-ingv-oe/data


## Citations

* https://github.com/se4ai2122-cs-uniba/Volcanic-eruption-prediction
* https://github.com/Pedro-Hdez/VolcanicEruptionPrediction
* https://github.com/chezka-sino/volcano-eruption
