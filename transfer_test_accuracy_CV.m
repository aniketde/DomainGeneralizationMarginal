function res_accu = transfer_test_accuracy_CV(X_test,Y_test,model,Wq,Q,Z_x,bw_kde1,bw_kde3,task_type)


new_vec = [bw_kde1*Z_x bw_kde3*X_test];
Z = [cos(new_vec*Wq) sin(new_vec*Wq)]/sqrt(Q);
Z_hat = sparse(Z);

[~, accuracy, ~] = predict(Y_test, Z_hat, model);
if strcmp(task_type,'regression')
    res_accu= accuracy(2);
elseif strcmp(task_type,'binary')
    res_accu = accuracy(1);
elseif strcmp(task_type,'multiclass')
    res_accu = accuracy(1);
end

end

