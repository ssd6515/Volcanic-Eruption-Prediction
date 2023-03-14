# Volcanic Eruption Prediction

* This repository holds an attempt to predict the time for the next volcanic eruption for each volcano from Kaggle challenge. 
* link - https://www.kaggle.com/competitions/predict-volcanic-eruptions-ingv-oe/overview/description

## Overview


 * **Definition of the tasks / challenge:**  The task is to predict when a volcano's next eruption will occur by analyzing a large geophysical dataset collected by sensors deployed on active volcanoes. 
 
 * **Approach:** The approach in this repository formulates the problem as a regression task, using deep neural networks as the model with the time series of seismic activity as input.  
 
 * **Summary of the performance achieved:** Our best model was able to predict the time for volcano's next eruption with the metric(MAE) of 11893938.230567.

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
  
  * Instances (Train, Test, Validation Split): data points: 1332. 482 volcano id for training, 730 for testing, 120 for validation

### Preprocessing / Clean up

Figure 1:

<img width="743" alt="g4" src="https://user-images.githubusercontent.com/89792366/225078119-83325f0a-b970-4013-ad4b-05b68478c6dd.png">

Figure 1: There are a lot of senosrs with missing data with sensor 2 being the highest. Thus, NAN values has been taken care of in preprocess for train_data and test_data for better modeling.

* Added the following features: 
     * sensor_1_nanmin-sensor_10_nanmin: The minimum value for 10 sensors in each csv file.                      
     * sensor_1_nanmax-sensor_10_nanmax: The maximum value for 10 sensors in each csv file.                                  
     * sensor_1_nanmedian-sensor_10_nanmedian: The median value for 10 sensors in each csv file.            
     * sensor_1_nanmean-sensor_10_nanmean: The mean value for 10 sensors in each csv file.   
     * sensor_1_nanstd-sensor_10_nanstd: The standard deviation value for 10 sensors in each csv file.       
     
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
  
  
   <img width="501" alt="plot" src="https://user-images.githubusercontent.com/89792366/225089612-4d731e58-28fe-4c60-aaae-86a34379e941.png"> 
  
  
  sensor_1_nanmean highest f-score. While sensor_10_nanstd has the lowest f-score.


* RandomForest Model:


  The model was created as follows:
  
  
   <img width="731" alt="m1" src="https://user-images.githubusercontent.com/89792366/225090629-96adc108-16da-4524-a390-5e5c141f608c.png">
  
  
  The model was trained with fit method:
  
  
   <img width="730" alt="t1" src="https://user-images.githubusercontent.com/89792366/225090682-80d8f99c-33fb-4d1a-a8ad-264dab85dfcf.png">



### Performance Comparison


* The performance metric is mean absolute error(MAE).

* Table:


   <img width="146" alt="t" src="https://user-images.githubusercontent.com/89792366/225091775-40508222-d355-4b78-9dac-9ec8cfddb3dd.png">

The  following plots show the difference between the difference between true Y_valid and predicted Y_pred from X_valid data.

* XGB plot:


  <img width="731" alt="g" src="https://user-images.githubusercontent.com/89792366/225092152-d2842b75-fe8a-474a-91e6-9c24a29bc752.png">
  
  
  
* RandomForest plot:


  <img width="736" alt="gf" src="https://user-images.githubusercontent.com/89792366/225092212-dd2e317e-81f5-4d85-8f60-557db02ff729.png">
  


### Conclusions

*  From the plots it is seen that the RandomForest model was the best among the 2 models as the predicted data was closer to the true value and condensed than the other models. But still, there is some variation from the true value of validation data in RandomForest model. Overall, the RandomForest model was effective only to the small samples of data used due to storage issues. This may have lead to larger MAE value than expected between 4000000-5000000.


### Future Work

* In future, I can make the models more effective in terms of getting accurate predictions. This can be done by deeply understanding the features used and advancing parameters used in models and using a better environment to process larger data. 

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
  * submission_rf.csv - file contaning item_cnt_month for the test data for randomforest model.
  * submission_xgb.csv - file contaning item_cnt_month for the test data for xgbregressor model.

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
