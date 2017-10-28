%clear
load('drdatarandom_1250_20160507_1332.mat')
datasets10 = cell(50,1);
for ii = 1:50
    x = X_tilde{ii}';
    y = Y{ii}';
    y(y==2) = 1;
    y(y==3) = 2;
     y(y==4) = 3;
    x = x(:,1:2);
    x = [x T{ii}'];
    x = (x - repmat(max(x),[length(x),1]))./repmat(min(x)-max(x),[length(x),1]);
    datasets10{ii}.testx = x;
    datasets10{ii}.testy = y(:,1);
end

% 
% for ii = 51:100
%     datasets1{ii,1} = datasets2{ii-50,1};
% end
% 
rand_p = randperm(500,500);
for ii = 1:100
    datasets{ii,1} = datasets{rand_p(ii)};
end

%datasets = [datasets1;datasets2;datasets3;datasets4;datasets5;datasets6;datasets7;datasets8;datasets9;datasets10];