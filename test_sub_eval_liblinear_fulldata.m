function res_accu = test_sub_eval_liblinear_fulldata(tube_sub,model,W,D)

tube_num = length(tube_sub);
res_accu = zeros(tube_num,1);
for jj=1:tube_num
    
    %Extracting test files
    Y_test = tube_sub{jj}.testy;
    X_test = [tube_sub{jj}.testx];
    size_test = size(X_test,1);
    %Big Kernel Matrix
    %K_test = zeros(size(X_train,1),size(X_test,1));
   Z = [cos(X_test*W) sin(X_test*W)]/sqrt(D);
    Z_hat = Z;
    Z_hat = sparse(Z_hat);
    
    [~, accuracy, ~] = predict(Y_test, Z_hat, model);
    
    disp(['test : ' num2str(jj) '   accuracy : ' num2str(accuracy(1))]);
    
    res_accu(jj) = accuracy(2);
    jj
    
    clear Z Z_hat Y_test X_test xse
end

end
