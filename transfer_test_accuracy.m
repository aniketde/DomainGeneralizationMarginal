function res_accu = transfer_test_accuracy(datasets,model,Wq,Q,Zx_full,bw_kde1,bw_kde3,task_type)

tube_num = length(datasets);
res_accu = zeros(tube_num,1);
for jj=1:tube_num
    %Extracting test files
    Y_test = datasets{jj}.testy;
    X_test = [datasets{jj}.testx];
    size_test = size(X_test,1);
    %Big Kernel Matrix
    %K_test = zeros(size(X_train,1),size(X_test,1));
    Zx = Zx_full(jj,:);
    e = ones(size_test,1);
    Z_x = kron(Zx,e);
    %X_train(:,size(X_train,2)+1:2*D) = 0;
    new_vec = [bw_kde1*Z_x bw_kde3*X_test];
    Z = [cos(new_vec*Wq) sin(new_vec*Wq)]/sqrt(Q);
    
    Z_hat = sparse(Z);
    
    [~, accuracy, ~] = predict(Y_test, Z_hat, model);
    
    disp(['test : ' num2str(jj) '   accuracy : ' num2str(accuracy(1))]);
    
    if strcmp(task_type,'regression')
        res_accu(jj) = accuracy(2);
    elseif strcmp(task_type,'binary')
        res_accu(jj) = accuracy(1);
    elseif strcmp(task_type,'multiclass')
        res_accu(jj) = accuracy(1);
    end
    
    clear new_vec Z_x Zx Z X_test Y_test Z_hat e
    jj
    
end

end
