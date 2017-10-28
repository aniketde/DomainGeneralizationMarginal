clear all;
close all;
clc;
rng('shuffle')
% THIS METHOD WORKS ONLY FOR FIVE FOLD CROSS VALIDATION. SO TUBE_SUB_TRAIN
% SHOULD BE ONLY MULTIPLE OF 5 (AND NOT 16,64 - MAKE IT 15,60, etc).
% TUBE_SUB_TEST can be anything.

%% load the training data based on the input
addpath(genpath('\\engin-labs.m.storage.umich.edu\aniketde\windat.v2\Desktop\sem 4 transfer learning\JMLR_experiments\imdb experiments\imdb_lgl\liblinear-1.95'))
str1 = 'tube_sub';
string_version = '_v1';
str = strcat(str1,string_version);
load(str);

N = length(datasets);
numberOfTrainingUser = 20; %5,10,15,20,25,30,35
% rand_perm = randperm(N,N);
% for ii = 1:N
%     tube_sub{ii} = tube_sub{rand_perm(ii)};
% end
% str1 = 'tube_sub_v10';
% str2  = '.mat';
% train_file = strcat(str1,str2); 
% save(train_file,'tube_sub');
%% Initialization
datasets_training_num = numberOfTrainingUser;
numberOfExamplesPerTask_grid = 100%ceil(logspace(1.3,2,10));

for nn = 1:length(numberOfExamplesPerTask_grid)
    
    numberOfExamplesPerTask = numberOfExamplesPerTask_grid(nn);
    bw_kde1_est =  390.6940;
    bw_kde2_est =   0.1758 ;
    bw_kde3_est = 575.4399;
    cost_est = 10;
    lambda =  1.2500e-05;
    bw_kde1_log =  logspace(-2,4,20);
    bw_kde2_log = 0.1758;
    bw_kde3_log =  575.4399;
    cost_log =  logspace(-1,1,10);
    L = 100;
    Q = 100;
    numb_iter = 5;
    
    
    datasets_training_num_array = (1:datasets_training_num)';
    tube_src_num = length(datasets_training_num_array);
    fold_cv = 5; %because of 5 fold cross-validation
    size_cv = datasets_training_num/fold_cv; %because of 5 fold cross-validation
    datasets_training_num_per_task = numberOfExamplesPerTask*ones(datasets_training_num,4);
    for ii = 1:datasets_training_num
        
        permrand = randperm(length(datasets{ii}.testy),numberOfExamplesPerTask);%1:N;
        datasets{ii}.x = datasets{ii}.testx(permrand,:);
        datasets{ii}.y = datasets{ii}.testy(permrand);
    end
    
    
    %% Data collect
    [X, Y, xse] = util_mrg_datasets(datasets, datasets_training_num_array, datasets_training_num_per_task);
    [x, y, xe,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,P_X_test,P_X_train] = util_mrg_datasets_CV(datasets, datasets_training_num_array, datasets_training_num_per_task,fold_cv,size_cv,X,Y);
    %util_mrg_datasets_CV
    
    %% estimation of bandwidth for pooling
    err = pooled_CV(bw_kde1_log,cost_log,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,Q);
    [~, idx] = min(err(:));
    [r1, r2] = ind2sub(size(err),idx);
    err(r1,r2)
    bw_kde1_est = bw_kde1_log(r1);
    cost_est = cost_log(r2);
    str1 = '_coordinate_ascent_cv_estimates_pooled_diffn_';
    str2 = num2str(numberOfExamplesPerTask);
    str3 = '.mat';
    str4= num2str(numberOfTrainingUser);
    str = strcat(str4,string_version,str1,str2,str3);
    save(str,'err','bw_kde1_log','bw_kde2_log','bw_kde3_log','cost_log','N','numberOfTrainingUser','fold_cv')
    bw_kde1_est_pooling = bw_kde1_est;
    cost_est_pooling  = cost_est;
    %% estimation of bandwidth for ktl first loop
    bw_kde1_log =  bw_kde1_est;
    bw_kde2_log = logspace(-2,4,20);
    bw_kde3_log =  logspace(-2,4,20);
    cost_log =  cost_est;
    err = transfer_CV(bw_kde1_log,bw_kde2_log,bw_kde3_log,cost_log,size_cv,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,L,Q,P_X_train,P_X_test,datasets);
    str1 = '_coordinate_ascent_cv_estimates_diff_patients_transfer_v1_diffn_';
    str2 = num2str(numberOfExamplesPerTask);
    str3 = '.mat';
   str4= num2str(numberOfTrainingUser);
    str = strcat(str4,string_version,str1,str2,str3);
    save(str,'err','bw_kde1_log','bw_kde2_log','bw_kde3_log','cost_log','N','numberOfTrainingUser','fold_cv')
    ani_err = reshape(err,[length(bw_kde2_log), length(bw_kde3_log)]);
    [~, idx] = min(err(:));
    [~, r1, r2] = ind2sub(size(err),idx);
    err(1,r1,r2);
    bw_kde2_est = bw_kde2_log(r1);
    bw_kde3_est = bw_kde3_log(r2);
    ani_err(r1,r2);
    
    
    %% estimation of bandwidth for ktl first loop
    bw_kde1_log =  logspace(-2,4,20);
    bw_kde2_log = bw_kde2_est;
    bw_kde3_log =  bw_kde3_est;
    cost_log =  logspace(-1,1,10);
    err = transfer_CV(bw_kde1_log,bw_kde2_log,bw_kde3_log,cost_log,size_cv,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,L,Q,P_X_train,P_X_test,datasets);
    str1 = '_coordinate_ascent_cv_estimates_diff_patients_transfer_v2_diffn_';
    str2 = num2str(numberOfExamplesPerTask);
    str3 = '.mat';
    str4= num2str(numberOfTrainingUser);
    str = strcat(str4,string_version, str1,str2,str3);
    save(str,'err','bw_kde1_log','bw_kde2_log','bw_kde3_log','cost_log','N','numberOfTrainingUser','fold_cv')
    err  = reshape(err,[length(bw_kde1_log), length(cost_log)]);
    [val_idx, idx] = min(err(:));
    [r1, r2] = ind2sub(size(err),idx);
    bw_kde1_est = bw_kde1_log(r1);
    cost_est = cost_log(r2);
    err(r1,r2);
    
    
    %% estimation of bandwidth for ktl first loop
    bw_kde1_log =  bw_kde1_est;
    bw_kde2_log = logspace(-2,4,20);
    bw_kde3_log =  logspace(-2,4,20);
    cost_log =  cost_est;
    err = transfer_CV(bw_kde1_log,bw_kde2_log,bw_kde3_log,cost_log,size_cv,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,L,Q,P_X_train,P_X_test,datasets);
    str1 = '_coordinate_ascent_cv_estimates_diff_patients_transfer_v3_diffn_';
    str2 = num2str(numberOfExamplesPerTask);
    str3 = '.mat';
    str4= num2str(numberOfTrainingUser);
    str = strcat(str4,string_version,str1,str2,str3);
    save(str,'err','bw_kde1_log','bw_kde2_log','bw_kde3_log','cost_log','N','numberOfTrainingUser','fold_cv')
    ani_err = reshape(err,[length(bw_kde2_log), length(bw_kde3_log)]);
    [~, idx] = min(err(:));
    [~, r1, r2] = ind2sub(size(err),idx);
    err(1,r1,r2);
    bw_kde2_est = bw_kde2_log(r1);
    bw_kde3_est = bw_kde3_log(r2);
    ani_err(r1,r2);
    
    
    %% comparison
    rand_perm_test = 36:42;
    L_grid = L;
    D = 100;
    for ii = datasets_training_num+1:rand_perm_test(end)
        datasets_training_num_per_task(ii,:) = length(datasets{ii}.testy)*ones(1,4);
    end
    for jj = datasets_training_num+1:rand_perm_test(end)
        permrand = randperm(length(datasets{jj}.testy),numberOfExamplesPerTask);%1:N;
        datasets{jj}.x = datasets{jj}.testx(permrand,:);
        datasets{jj}.y = datasets{jj}.testy(permrand);
    end
    
    [res_avg_test, ~,res_avg_train,~] = pooled_transfer_comparison(datasets_training_num,bw_kde1_est,bw_kde2_est,bw_kde3_est,cost_est,rand_perm_test,L_grid,Q,D,numb_iter,datasets,datasets_training_num_per_task);
    str1 = '_liblinear_ktl_diffn_';
    str2 = '.mat';
    str3= num2str(numberOfTrainingUser);
    str = strcat(str3,string_version,str1,str2);
    res_avg_test_loop(nn) = res_avg_test;
    res_avg_train_loop(nn) = res_avg_train;
    save(str,'res_avg_test_loop','res_avg_train_loop')
    
    [~, res_avg_test_pooled,~,res_avg_train_pooled] = pooled_transfer_comparison(datasets_training_num,bw_kde1_est_pooling,bw_kde2_est,bw_kde3_est,cost_est_pooling,rand_perm_test,L_grid,Q,D,numb_iter,datasets,datasets_training_num_per_task);
    str1 = '_liblinear_pooling_diffn_';
    str2 = '.mat';
    str3= num2str(numberOfTrainingUser);
    str = strcat(str3,string_version,str1,str2);
    res_avg_test_pooled_loop(nn) = res_avg_test_pooled;
    res_avg_train_pooled_loop(nn) = res_avg_train_pooled;
     save(str,'res_avg_test_pooled_loop','res_avg_train_pooled_loop')
    
    
end