function [x, y, xe] = util_mrg_datasets(datasets, datasets_training_num_array, datasets_training_num_per_task)

tube_sub_size = datasets_training_num_per_task(datasets_training_num_array,1);
tube_sub_idx = cumsum([0; tube_sub_size]);

x = zeros(sum(tube_sub_size), size(datasets{1}.x,2));
xe = zeros(size(x,1), size(x,2)+1);
y = zeros(size(x,1),1);

for ii=1:length(datasets_training_num_array)
    
    idx = tube_sub_idx(ii)+1:tube_sub_idx(ii+1);
    x(idx,:) = datasets{datasets_training_num_array(ii)}.x;
    xe(idx,:) = [datasets{datasets_training_num_array(ii)}.x, datasets_training_num_array(ii)*ones(length(idx),1)];
    y(idx) = datasets{datasets_training_num_array(ii)}.y;
    
end

end

