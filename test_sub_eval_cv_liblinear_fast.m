function res_accu = test_sub_eval_cv_liblinear_fast(X_test,Y_test,model,Wq,Q,Z_x,bw_kde1,bw_kde3)


new_vec = [bw_kde1*Z_x bw_kde3*X_test];
Z = [cos(new_vec*Wq) sin(new_vec*Wq)]/sqrt(Q);
Z_hat = sparse(Z);

[~, accuracy, ~] = predict(Y_test, Z_hat, model);
res_accu = accuracy(2);
end

