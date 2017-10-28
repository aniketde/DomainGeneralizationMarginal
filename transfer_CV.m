function [err_iii] = transfer_CV(bw_kde1_log,bw_kde2_log,bw_kde3_log,cost_log,size_cv,numb_iter,X_test_cv,Y_test_cv,X_train_cv,Y_train_cv,fold_cv,L,Q,P_X_train_cv,P_X_test_cv,datasets,task_type)

err_iii =  zeros(length(bw_kde1_log),length(bw_kde2_log),length(bw_kde3_log),length(cost_log));
for bb1 = 1:length(bw_kde1_log)
    bw_kde1 = bw_kde1_log(bb1);
    for bb2 = 1:length(bw_kde2_log)
        bw_kde2 = bw_kde2_log(bb2);
        for bb3 = 1:length(bw_kde3_log)
            bw_kde3 = bw_kde3_log(bb3);
            for cc = 1:length(cost_log)
                cost = cost_log(cc);
                tic
                for iii = 1:fold_cv
                    
                    for rand_num = 1:numb_iter
                        X_test =X_test_cv{iii};
                        Y_test = Y_test_cv{iii};
                        X_train = X_train_cv{iii}; %training points
                        Y_train = Y_train_cv{iii}; %training points
                        P_X_train = P_X_train_cv{iii};
                        P_X_test = P_X_test_cv{iii};
                        
                        Wl = randn(size(X_train,2),L)/(bw_kde2);
                        
                        Zx = zeros((fold_cv-1)*size_cv,2*L);
                        Z_x = zeros(size(X_train,1),2*L);
                        jj = 0;
                        for ii=1:((fold_cv-1)*size_cv)
                            xi = datasets{P_X_train(ii)}.x;%X_train(size_train*(ii-1)+1:size_train*ii,:);
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
                        
                        %% initialization for testing
                        
                        %%testing
                        Zx = zeros((fold_cv-4)*size_cv,2*L);
                        Z_x = zeros(size(X_test,1),2*L);
                        jj = 0;
                        for ii=1:((fold_cv-4)*size_cv)
                            %xi = X_test(size_train*(ii-1)+1:size_train*ii,:);
                            %Zx(ii,:) = [sum(cos(xi*Wl)) sum(sin(xi*Wl))]/(sqrt(L)*length(xi));
                            xi = datasets{P_X_test(ii)}.x;%X_train(size_train*(ii-1)+1:size_train*ii,:);
                            Zx(ii,:) = [sum(cos(xi*Wl)) sum(sin(xi*Wl))]/(sqrt(L)*length(xi));
                            e = ones(length(xi),1);
                            ind_jj = jj+1:jj+length(xi);
                            Z_x(ind_jj,:) = kron(Zx(ii,:),e);
                            jj = jj + length(xi);
                        end
                        res_accu_numb_iter(rand_num) = transfer_test_accuracy_CV(X_test,Y_test,model,Wq,Q,Z_x,bw_kde1,bw_kde3,task_type);
                        clear X_test X_train Y_train P_X_train P_X_test Zx Z_x new_vec Z Z_hat
                    end
                    
                    res_accu(iii) = mean(res_accu_numb_iter);
                    if strcmp(task_type,'regression')
                        mtl_kderbf1_err(iii) = res_accu(iii);%(100-res_accu(iii));
                    elseif strcmp(task_type,'binary')
                        mtl_kderbf1_err(iii) =(100-res_accu(iii));
                    elseif strcmp(task_type,'multiclass')
                        mtl_kderbf1_err(iii) =(100-res_accu(iii));
                    end
                    
                end
                toc
                err_iii(bb1,bb2,bb3,cc) = mean(mtl_kderbf1_err);
            end
        end
    end
end
