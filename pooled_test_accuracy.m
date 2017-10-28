function res_accu = pooled_test_accuracy(datasets,model,W,D,task_type)

tube_num = length(datasets);
res_accu = zeros(tube_num,1);
for jj=1:tube_num
    
    %Extracting test files
    Y_test = datasets{jj}.testy;
    X_test = [datasets{jj}.testx];
    size_test = size(X_test,1);
    %Big Kernel Matrix
    %K_test = zeros(size(X_train,1),size(X_test,1));
    Z = [cos(X_test*W) sin(X_test*W)]/sqrt(D);
    Z_hat = Z;
    Z_hat = sparse(Z_hat);
    
    [~, accuracy, ~] = predict(Y_test, Z_hat, model);
    
    disp(['test : ' num2str(jj) '   accuracy : ' num2str(accuracy(1))]);
    
    if strcmp(task_type,'regression')
        res_accu(jj) = accuracy(2);
    elseif strcmp(task_type,'binary')
        res_accu(jj) = accuracy(1);
    elseif strcmp(task_type,'multiclass')
        res_accu(jj) = accuracy(1);
    end
    
    clear Z Z_hat Y_test X_test xse
end

end
