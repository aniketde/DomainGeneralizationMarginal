clear all
clc

NumberOfPointsPerTask = 100;
Number_Of_Tasks = 3;
factor = 0.2;
TensorOfTasks = getTaskData(NumberOfPointsPerTask, Number_Of_Tasks, factor);
[MatrixOfLabels, PositiveExamples, NegativeExamples]  = getLabelsAndClassBasedData(TensorOfTasks,NumberOfPointsPerTask, Number_Of_Tasks);

Elipse = calculateEllipse(0, 0, 1, factor, 0, 1000);
%{
plot(Elipse(: , 1),Elipse(:,2),'g', 'LineWidth', 5)
hold on; 
plot(PositiveExamples(1,:,1),PositiveExamples(1,:,2),'r*')
hold on; 
plot(NegativeExamples(1,:,1),NegativeExamples(1,:,2),'b*')
%}


numberExamplesPerTaskForTraining       = 25;
numberOfPositiveExamplesForTraining = floor(numberExamplesPerTaskForTraining * 0.5);
numberOfNegativeExamplesForTraining = numberExamplesPerTaskForTraining - numberOfPositiveExamplesForTraining;

initialCaseIndex    = ones(numberExamplesPerTaskForTraining,1);
CaseIndexes         = zeros(numberExamplesPerTaskForTraining,1);
for i=1:1:Number_Of_Tasks-2
    CaseIndexes = [ CaseIndexes; i*initialCaseIndex ];
end


rotation_angles = calculateRotationAngles(-1*pi,0, Number_Of_Tasks);
for taskCounter = 1:1:Number_Of_Tasks
    matrix = reshape(PositiveExamples(taskCounter,:,:),size(PositiveExamples,2),size(PositiveExamples,3));
    RotatedPositiveExamples = rotateCellTensor( matrix,rotation_angles(taskCounter));
    clear matrix;
    matrix = reshape(NegativeExamples(taskCounter,:,:),size(NegativeExamples,2),size(NegativeExamples,3));
    RotatedNegativeExamples = rotateCellTensor( matrix,rotation_angles(taskCounter));
    clear matrix;
    RotatedElipse = rotateMatrix(Elipse, rotation_angles(taskCounter));
    TensorOfRotatedPositiveExamples{taskCounter} = RotatedPositiveExamples;
    TensorOfRotatedNegativeExamples{taskCounter} = RotatedNegativeExamples;
    TensorOfRotatedElipse{taskCounter} = RotatedElipse;
    
  
    for exampleCounter = 1:1:numberOfPositiveExamplesForTraining
        TensorOfTrainingDataPerTask(taskCounter,exampleCounter,1) = RotatedPositiveExamples(exampleCounter , 1 );
        TensorOfTrainingDataPerTask(taskCounter,exampleCounter,2) = RotatedPositiveExamples(exampleCounter , 2 ); 
    end
    
    for exampleCounter = 1:1:numberOfNegativeExamplesForTraining
        TensorOfTrainingDataPerTask(taskCounter,numberOfPositiveExamplesForTraining+exampleCounter, 1) = RotatedPositiveExamples(exampleCounter,1);
        TensorOfTrainingDataPerTask(taskCounter,numberOfPositiveExamplesForTraining+exampleCounter, 2) = RotatedPositiveExamples(exampleCounter,2);   
    end
    
    clear RotationMatrix;
    clear RotatedPositiveExamples;
    clear RotatedNegativeExamples;
    clear RotatedElipse;
end
for numberOfTasks = 1:1:Number_Of_Tasks
    plot(TensorOfRotatedElipse(numberOfTasks,:,1),TensorOfRotatedElipse(numberOfTasks,:,2),'g', 'LineWidth', 0.4); 
    hold on; 
    plot(TensorOfRotatedPositiveExamples(numberOfTasks,:,1),TensorOfRotatedPositiveExamples(numberOfTasks,:,2),'r*');
    hold on;
    plot(TensorOfRotatedNegativeExamples(numberOfTasks,:,1),TensorOfRotatedNegativeExamples(numberOfTasks,:,2),'b*')
end

for testTaskCounter = 1:1:Number_Of_Tasks
    for trainingTaskCounter = 1:1:Number_Of_Tasks
        if(testTaskCounter == trainingTaskCounter)% Get Tes Data
             TensorOfTestData(testTaskCounter,:,:) = [TensorOfRotatedPositiveExamples(testTaskCounter,:,:), TensorOfRotatedNegativeExamples(testTaskCounter,:,:)];
        else% construct training data
            
             TensorOfTrainingData(testTaskCounter,:,:) = [TensorOfTrainingData(testTaskCounter,:,:),  TensorOfTrainingDataPerTask(taskCounter,:,:)];
        end
        
    end
end












