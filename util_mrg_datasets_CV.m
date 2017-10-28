function [x, y, xe,X_test,Y_test,X_train,Y_train,P_X_test,P_X_train] = util_mrg_datasets_CV(datasets, datasets_training_num_array, datasets_training_num_per_task,fold_cv,size_cv,X,Y)

datasets_training_num_per_task = datasets_training_num_per_task(datasets_training_num_array,1);
tube_sub_idx = cumsum([0; datasets_training_num_per_task]);
%tube_sub_size_cv = tubesize(tubeid,1)/(tube_sub_size(1)/size_cv);
%tube_sub_idx_cv = cumsum([0; tube_sub_size_cv]);

x = zeros(sum(datasets_training_num_per_task), size(datasets{1}.x,2));
xe = zeros(size(x,1), size(x,2)+1);
y = zeros(size(x,1),1);
X_test=cell(fold_cv,1);
Y_test=cell(fold_cv,1);
X_train=cell(fold_cv,1);
Y_train=cell(fold_cv,1);
P_X_test=cell(fold_cv,1);
P_X_train = cell(fold_cv,1);
%numb_valid = tube_sub_size(1)*size_cv;
prev_numb_valid = 0;
for ii=1:fold_cv
    numb_valid = prev_numb_valid+ sum(datasets_training_num_per_task((ii-1)*size_cv+1:ii*size_cv));
    P_X_test{ii} = datasets_training_num_array((ii-1)*size_cv+1:ii*size_cv);
    X_test{ii} = X(prev_numb_valid+1:numb_valid,:); %validation points
    Y_test{ii} = Y(prev_numb_valid+1:numb_valid,:); %validation points
    P_X_train{ii} = removerows(datasets_training_num_array,'ind',(ii-1)*size_cv+1:ii*size_cv);
    [X_train{ii},PS] = removerows(X,'ind',prev_numb_valid+1:numb_valid); %training points
    [Y_train{ii},PS] = removerows(Y,'ind',prev_numb_valid+1:numb_valid); %training points
    prev_numb_valid = numb_valid;
  
end

