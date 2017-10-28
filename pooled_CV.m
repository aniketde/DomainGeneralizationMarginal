function [err_iii] = pooled_CV(bw_kde1_log,cost_log,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,Q,task_type)

err_iii =  zeros(length(bw_kde1_log),length(cost_log));
for bb1 = 1:length(bw_kde1_log)
    bw_kde1 = bw_kde1_log(bb1);
    for cc = 1:length(cost_log)
        cost = cost_log(cc);
        tic
        for iii = 1:fold_cv
            
            for rand_num = 1:numb_iter
                X_test =X_test_cv{iii};
                Y_test = Y_test_cv{iii};
                X_train = X_train_cv{iii}; %training points
                Y_train = Y_train_cv{iii}; %training points
                Wq = randn(size(X_train,2),Q)/(bw_kde1);
                Z = [cos(X_train*Wq) sin(X_train*Wq)]/sqrt(Q);
                Z_hat = sparse(Z);
                if strcmp(task_type,'regression')
                    model = train(Y_train, Z_hat, ['-s 11 -q -c ' num2str(cost)]);
                elseif strcmp(task_type,'binary')
                    model = train(Y_train, Z_hat, ['-s 1 -q -c ' num2str(cost)]);
                elseif strcmp(task_type,'multiclass')
                    model = train(Y_train, Z_hat, ['-s 4 -q -c ' num2str(cost)]);
                end
                %%testing
                
                Z = [cos(X_test*Wq) sin(X_test*Wq)]/sqrt(Q);
                Z_hat = sparse(Z);
                [~, accuracy, ~] = predict(Y_test, Z_hat, model);
                if strcmp(task_type,'regression')
                    res_accu_numb_iter(rand_num) = accuracy(2);
                elseif strcmp(task_type,'binary')
                    res_accu_numb_iter(rand_num) = accuracy(1);
                elseif strcmp(task_type,'multiclass')
                    res_accu_numb_iter(rand_num) = accuracy(1);
                end
            end
            res_accu(iii) = mean(res_accu_numb_iter);
            if strcmp (task_type,'regression')
                mtl_kderbf1_err(iii) = res_accu(iii);%(100-res_accu(iii));
            elseif strcmp(task_type,'binary')
                mtl_kderbf1_err(iii) =(100-res_accu(iii));
            elseif strcmp(task_type,'multiclass')
                mtl_kderbf1_err(iii) =(100-res_accu(iii));
            end
        end
        toc
        err_iii(bb1,cc) = mean(mtl_kderbf1_err);
    end
end


