function [res_avg_test,res_avg_train,res_avg_test_pooled,res_avg_train_pooled] = cross_validation_pooling_transfer(useAllExamples,numberOfExamplesPerTask,numberOfTrainingUser,datasets,L,Q,D,cross_val,rand_perm_test,numb_iter,task_type,bw1_est,bw2_est,bw3_est,c_est,bw1_est_pooling,c_est_pooling)


%% Initialization
datasets_training_num = numberOfTrainingUser;
datasets_training_num_array = (1:datasets_training_num)';
if useAllExamples == 0
    datasets_training_num_per_task = numberOfExamplesPerTask*ones(datasets_training_num,4);
    for ii = 1:datasets_training_num
        permrand = randperm(length(datasets{ii}.testy),numberOfExamplesPerTask);%1:N;
        datasets{ii}.x = datasets{ii}.testx(permrand,:);
        datasets{ii}.y = datasets{ii}.testy(permrand);
    end
else
    datasets_training_num_per_task = zeros(datasets_training_num,4);
    for ii = 1:datasets_training_num
        datasets{ii}.x = datasets{ii}.testx;
        datasets{ii}.y = datasets{ii}.testy;
        datasets_training_num_per_task(ii,:) = repmat(length(datasets{ii}.testy),[1,4]);
    end
end


if cross_val == 1
    bw_kde1_log = logspace(-2,4,20);
    cost_log =  logspace(-1,1,10);
    fold_cv = 5; %because of 5 fold cross-validation
    size_cv = datasets_training_num/fold_cv; %because of 5 fold cross-validation
    %% Data collect
    [X, Y, ~] = util_mrg_datasets(datasets, datasets_training_num_array, datasets_training_num_per_task);
    [~, ~, ~,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,P_X_test,P_X_train] = util_mrg_datasets_CV(datasets, datasets_training_num_array, datasets_training_num_per_task,fold_cv,size_cv,X,Y);
    %util_mrg_datasets_CV
    
    %% estimation of bandwidth for pooling
    err = pooled_CV(bw_kde1_log,cost_log,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,Q,task_type);
    [~, idx] = min(err(:));
    [r1, r2] = ind2sub(size(err),idx);
    err(r1,r2)
    bw_kde1_est = bw_kde1_log(r1);
    cost_est = cost_log(r2);
    bw_kde1_est_pooling = bw_kde1_est;
    cost_est_pooling  = cost_est;
    %% estimation of bandwidth for ktl first loop
    bw_kde1_log =  bw_kde1_est;
    bw_kde2_log = logspace(-2,4,20);
    bw_kde3_log =  logspace(-2,4,20);
    cost_log =  cost_est;
    err = transfer_CV(bw_kde1_log,bw_kde2_log,bw_kde3_log,cost_log,size_cv,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,L,Q,P_X_train,P_X_test,datasets,task_type);
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
    err = transfer_CV(bw_kde1_log,bw_kde2_log,bw_kde3_log,cost_log,size_cv,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,L,Q,P_X_train,P_X_test,datasets,task_type);
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
    err = transfer_CV(bw_kde1_log,bw_kde2_log,bw_kde3_log,cost_log,size_cv,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,L,Q,P_X_train,P_X_test,datasets,task_type);
    ani_err = reshape(err,[length(bw_kde2_log), length(bw_kde3_log)]);
    [~, idx] = min(err(:));
    [~, r1, r2] = ind2sub(size(err),idx);
    err(1,r1,r2);
    bw_kde2_est = bw_kde2_log(r1);
    bw_kde3_est = bw_kde3_log(r2);
    ani_err(r1,r2);
    
else
    
    bw_kde1_est = bw1_est;
    bw_kde2_est = bw2_est;
    bw_kde3_est = bw3_est;
    cost_est = c_est;
    bw_kde1_est_pooling = bw1_est_pooling;
    cost_est_pooling = c_est_pooling;
end
%% comparison

for ii = datasets_training_num+1:rand_perm_test(end)
    datasets_training_num_per_task(ii,:) = length(datasets{ii}.testy)*ones(1,4);
end

% for jj = datasets_training_num+1:rand_perm_test(end)
%     permrand = randperm(length(datasets{jj}.testy),numberOfExamplesPerTask);%1:N;
%     datasets{jj}.x = datasets{jj}.testx(permrand,:);
%     datasets{jj}.y = datasets{jj}.testy(permrand);
% end

if useAllExamples == 0
    for jj = datasets_training_num+1:rand_perm_test(end)
        permrand = randperm(length(datasets{jj}.testy),numberOfExamplesPerTask);%1:N;
        datasets{jj}.x = datasets{jj}.testx(permrand,:);
        datasets{jj}.y = datasets{jj}.testy(permrand);
    end
else
    for jj = datasets_training_num+1:rand_perm_test(end)
        datasets{jj}.x = datasets{jj}.testx;
        datasets{jj}.y = datasets{jj}.testy;
        datasets_training_num_per_task(jj,:) = repmat(length(datasets{jj}.testy),[1,4]);
    end
end


[res_avg_test, ~,res_avg_train,~] = pooled_transfer_comparison(datasets_training_num,bw_kde1_est,bw_kde2_est,bw_kde3_est,cost_est,rand_perm_test,L,Q,D,numb_iter,datasets,datasets_training_num_per_task,task_type);
[~, res_avg_test_pooled,~,res_avg_train_pooled] = pooled_transfer_comparison(datasets_training_num,bw_kde1_est_pooling,bw_kde2_est,bw_kde3_est,cost_est_pooling,rand_perm_test,L,Q,D,numb_iter,datasets,datasets_training_num_per_task,task_type);

