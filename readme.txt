Domain Generalization by Marginal Transfer Learning
 
This compares marginal predictor method [1] with pooling method.
It uses kernel approximation and then liblinear as a solver for speed up.
This code gives comparison of pooling and transfer learning for regression and binary classification. 

%% Instructions - 
1) Download liblinear library and put it in the main workspace. 
2) Compile the library from matlab folder inside it. 
3) For demo, run demo_transfer_pooling. 

%% Files
1) demo_transfer_pooling = Gives demo for both regression and binary classification
2) cross_validation_pooling_transfer = This is a main file and automatically chooses the optimization parameter plus kernel bandwidths. One can also give their own parameters. After choosing bandwiths it gives error comparison of pooling and transfer learning.
3) util_mrg_datasets = merge all datasets 
4) util_mrg_datasets_CV = merge and split training datasets into training and validation sets 
5) pooled_CV = Cross Validation for pooling method
6) transfer_CV = Cross Validation for transfer learning method
7) transfer_test_accuracy_CV = Calculates the accuracy on validtion set for transfer learning method
8) pooled_transfer_comparison = After parameters are set, this file calcualtes the actual training and test error and does comparison
9) pooled_test_accuracy = Calculates error for pooling method
10) transfer_test_accuracy = Calculates error for transfer learning method
11) synth_binary_data = synthtic data for binary classification

%% Demo 
run demo_transfer_pooling. 
This compares marginal predictor method [1] with pooling method.
It uses kernel approximation and then liblinear as a solver for speed up
input - train and test datasets = datasets in cell format.
      - datasets{i}.testx contains all features and Dataset{i}.testy
      contains all labels. Don't get mislead by testx name. It contains
      all datapoints. 
      - numberOfTrainingUser = number of datasets that are used for training
      In datasets First numberOfTrainingUser are considered as training
      datasets
      - numberOfExamplesPerTask 
      - rand_perm_test - datasets to be used as test datasets
      - task_type - it could be 'regression' or 'binary'
      - cross_val  - 1 if you want to do cross val otherwise 0
      - L,Q,D - number of random fourier features to approximate kernel
output -  res_avg_test = test error using marginal predictors method 
       - res_avg_train = train error using marginal predictors method
       - res_avg_test_pooled = test error using pooling method
       - res_avg_train_pooled = train error using pooling method
       - In case of regression - squared error 
       - In case of binary classification - % 0-1 error
Warning - THIS METHOD WORKS ONLY FOR FIVE FOLD CROSS VALIDATION. 
           SO numberOfTrainingUser SHOULD BE ONLY MULTIPLE OF 5 

