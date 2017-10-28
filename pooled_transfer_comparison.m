function [res_avg_test, res_avg_test_pooled,res_avg_train,res_avg_train_pooled] = pooled_transfer_comparison(datasets_training_num,bw_kde1,bw_kde2,bw_kde3,cost,rand_perm_test,L_grid,Q,D,numb_iter,datasets,datasets_training_num_per_task,task_type )
%% Initialization
tube_num = length(datasets);
rand_perm = 1:datasets_training_num;
tube_src = (rand_perm)';
tube_src_num = length(tube_src);
res_avg = zeros(size(numb_iter,length(L_grid)));
res_cell = cell(size(numb_iter,length(L_grid)));
res_test = res_avg;
res_train = res_avg;
res_accu_all = res_cell;
res_cell_pooled = res_cell;
res_test_pooled = res_avg;
res_train_pooled = res_avg;
for D_i = 1:length(L_grid)
    
    for rand_numb = 1:numb_iter
        
        [X_train, Y_train, xse] = util_mrg_datasets(datasets, tube_src, datasets_training_num_per_task);
        
        tic
        L = ceil(L_grid(D_i));
        Wl = randn(size(X_train,2),L)/(bw_kde2);
        Zx = zeros(tube_src_num,2*L);
        Z_x = zeros(size(X_train,1),2*L);
        jj = 0;
        for ii=1:tube_src_num
            xi = datasets{rand_perm(ii)}.x;
            Zx(ii,:) = [sum(cos(xi*Wl)) sum(sin(xi*Wl))]/(sqrt(L)*length(xi));
            e = ones(length(xi),1);
            ind_jj = jj+1:jj+length(xi);
            Z_x(ind_jj,:) = kron(Zx(ii,:),e);
            jj = jj + length(xi);
        end
        
        new_vec = [bw_kde1*Z_x bw_kde3*X_train];
        Wq = randn(size(new_vec,2),Q)/(bw_kde1*bw_kde3);
        
        Z = [cos(new_vec*Wq) sin(new_vec*Wq)]/sqrt(Q);
        Z_hat = sparse(Z);
        
        if strcmp(task_type,'regression')
            model = train(Y_train, Z_hat, ['-s 11 -q -c ' num2str(cost)]);
        elseif strcmp(task_type,'binary')
            model = train(Y_train, Z_hat, ['-s 1 -q -c ' num2str(cost)]);
        elseif strcmp(task_type,'multiclass')
            model = train(Y_train, Z_hat, ['-s 4 -q -c ' num2str(cost)]);
        end
        
        
        
        
        Wd = randn(size(X_train,2),D)/(bw_kde1);
        Z_pooled = [cos(X_train*Wd) sin(X_train*Wd)]/sqrt(D);
        Z_hat_pooled = Z_pooled;
        Z_hat_pooled = sparse(Z_hat_pooled);
        
        if strcmp(task_type,'regression')
            model_pooled = train(Y_train, Z_hat_pooled, ['-s 11 -q -c ' num2str(cost)]);
        elseif strcmp(task_type,'binary')
            model_pooled = train(Y_train, Z_hat_pooled, ['-s 1 -q -c ' num2str(cost)]);
        elseif strcmp(task_type,'multiclass')
            model_pooled = train(Y_train, Z_hat_pooled, ['-s 4 -q -c ' num2str(cost)]);
        end
        
        
        clear Z Z_hat new_vec e Z_x xs xse xt ys yt
        
        %% Task similarity for everything
        Zx_full = zeros(tube_num,2*L);
        
        for ii=1:tube_num
            xi = datasets{(ii)}.x;%X_train(size_train*(ii-1)+1:size_train*ii,:);
            Zx_full(ii,:) = [sum(cos(xi*Wl)) sum(sin(xi*Wl))]/(sqrt(L)*length(xi));
        end
        
        %% initialization for testing
        res_accu = transfer_test_accuracy(datasets,model,Wq,Q,Zx_full,bw_kde1,bw_kde3,task_type);
        if strcmp(task_type,'regression')
            mtl_kderbf1_err = res_accu;%(100-res_accu);
        elseif strcmp(task_type,'binary')
            mtl_kderbf1_err = (100-res_accu);
        elseif strcmp(task_type,'multiclass')
            mtl_kderbf1_err = (100-res_accu);
        end
        
        res = mtl_kderbf1_err;
        res_cell{rand_numb,D_i} = res;
        res_test(rand_numb,D_i) = mean(res(rand_perm_test));
        res_train(rand_numb,D_i) = mean(res(rand_perm));
        
        
        %% initialization for testing
        
        res_accu_pooled = pooled_test_accuracy(datasets,model_pooled,Wd,D,task_type);
        res_accu_all{rand_numb,D_i} = res_accu_pooled;
        %%
        % result figure
        if strcmp(task_type,'regression')
            mtl_kderbf1_err_pooled = res_accu_pooled;%(100-res_accu_pooled);
        elseif strcmp(task_type,'binary')
            mtl_kderbf1_err_pooled =(100-res_accu_pooled);
        elseif strcmp(task_type,'multiclass')
            mtl_kderbf1_err_pooled =(100-res_accu_pooled);
        end
        
        
        
        res_pooled = mtl_kderbf1_err_pooled;
        res_cell_pooled{rand_numb,D_i} = res_pooled;
        res_test_pooled(rand_numb,D_i) = mean(res_pooled(rand_perm_test));
        res_train_pooled(rand_numb,D_i) = mean(res_pooled(rand_perm));
        
        
    end
    
end

res_avg_test  = mean(res_test);
res_avg_train  = mean(res_train);
res_avg_test_pooled  = mean(res_test_pooled);
res_avg_train_pooled  = mean(res_train_pooled);
