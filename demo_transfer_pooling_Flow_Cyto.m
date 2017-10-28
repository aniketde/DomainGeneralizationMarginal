clear all;
close all;
clc;
rng('shuffle')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This compares marginal predictor method [1] with pooling method.
% It uses kernel approximation and then liblinear as a solver for speed up
% input - train and test datasets = datasets in cell format.
%       - datasets{i}.testx contains all features and Dataset{i}.testy
%       contains all labels. Don't get mislead by testx name. It contains
%       all datapoints. 
%       - numberOfTrainingUser = number of datasets that are used for training
%       In datasets First numberOfTrainingUser are considered as training
%       datasets
%       - numberOfExamplesPerTask 
%       - rand_perm_test - datasets to be used as test datasets
%       - task_type - it could be 'regression' or 'binary'
%       - cross_val  - 1 if you want to do cross val otherwise 0
%       - L,Q,D - number of random fourier features to approximate kernel
% output -  res_avg_test = test error using marginal predictors method 
%        - res_avg_train = train error using marginal predictors method
%        - res_avg_test_pooled = test error using pooling method
%        - res_avg_train_pooled = train error using pooling method
%        - In case of regression - squared error 
%        - In case of binary classification - % 0-1 error
% Warning - THIS METHOD WORKS ONLY FOR FIVE FOLD CROSS VALIDATION. 
%            SO numberOfTrainingUser SHOULD BE ONLY MULTIPLE OF 5 
% Author - Aniket Deshmukh, Clayton  Scott
% [1] Blanchard, Gilles, Gyemin Lee, and Clayton Scott. "Generalizing from several related classification tasks to a new unlabeled sample." 
% In Advances in neural information processing systems, pp. 2178-2186. 2011.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% load the training data based on the input
current_path = pwd;
liblinear_path = strcat(current_path,'\liblinear-1.95');
addpath(genpath(liblinear_path))
load flow_cyto_multi_35 % synth_parkinson_regression_data

task_type = 'multiclass'; % It could be either 'binary' or 'regression'
numberOfTrainingUser = 30; %5,10,15,20,25,30,35, should be multuple of 5 and first (1:numberOfTrainingUser) will be selected as training datasets
useAllExamples = 0;
numberOfExamplesPerTask = 1000;

    
L = 100;
Q = 100;
D = 100;
numb_iter = 5;
rand_perm_test = 31:35; %  test datasets 
cross_val = 1; % 1 if you want to do cross validation, 0 if you don't want to
% If you don't want to do cross validation then input all parameters here. 
if cross_val == 0
   bw1_est = 100; 
   bw2_est = 100;
   bw3_est = 100;
   c_est = 1;
   bw1_est_pooling = 100;
   c_est_pooling = 1; 
elseif cross_val == 1
    bw1_est = 100;
   bw2_est = 100;
   bw3_est = 100;
   c_est = 1;
   bw1_est_pooling = 100;
   c_est_pooling = 1; 
end

[res_avg_test,res_avg_train,res_avg_test_pooled,res_avg_train_pooled] = cross_validation_pooling_transfer(useAllExamples,numberOfExamplesPerTask,numberOfTrainingUser,datasets,L,Q,D,cross_val,rand_perm_test,numb_iter,task_type,bw1_est,bw2_est,bw3_est,c_est,bw1_est_pooling,c_est_pooling);
% if strcmp(task_type,'binary')
%     res_avg_test = 100 - res_avg_test;
%     res_avg_train = 100 - res_avg_train;
%     res_avg_test_pooled = 100- res_avg_test_pooled;
%     res_avg_train_pooled = 100- res_avg_train_pooled;
% end
save('flow_cyto_multi_35_1000_results','res_avg_test','res_avg_train','res_avg_test_pooled','res_avg_train_pooled')
    