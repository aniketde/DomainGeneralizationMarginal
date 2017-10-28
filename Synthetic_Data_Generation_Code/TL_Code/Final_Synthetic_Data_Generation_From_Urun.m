clear all
close all
nb_Tasks = 256
nb_TrnExaPerTask = 1000
random = 1
repeat_Number = 2

[TensorOfTrainingDataPerTask_Matrix,TrainingLabelsPerTask] = ToyData_New_Version_cell(nb_Tasks, nb_TrnExaPerTask, random, repeat_Number);
figure
datasets = cell(nb_Tasks,1);
for ii = 1:nb_Tasks
    x = reshape(TensorOfTrainingDataPerTask_Matrix(ii,:,:), [nb_TrnExaPerTask,2] );
    y = TrainingLabelsPerTask{ii};
    datasets{ii}.testx = x;
    datasets{ii}.testy = y;
    hold on
    scatter(x(:,1),x(:,2),[],y)
end

    