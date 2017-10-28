function ToyData_New_Version_cell(nb_Tasks, nb_TrnExaPerTask, random, repeat_Number)
Number_Of_Tasks = nb_Tasks;
numberExamplesPerTaskForTraining = nb_TrnExaPerTask;
NumberOfPointsPerTask = 100 + numberExamplesPerTaskForTraining; 
Run = random; 
numberOfPositiveExamplesForTraining = floor(numberExamplesPerTaskForTraining * 0.5);
numberOfNegativeExamplesForTraining = numberExamplesPerTaskForTraining - numberOfPositiveExamplesForTraining;

factor = 0.2;
TensorOfTasks = getTaskData(NumberOfPointsPerTask, Number_Of_Tasks, factor);
[PositiveExamples, NegativeExamples]  = getLabelsAndClassBasedData(TensorOfTasks,NumberOfPointsPerTask, Number_Of_Tasks);

Elipse = calculateEllipse(0, 0, 1, factor, 0, 1000);

%{
plot(Elipse(: , 1),Elipse(:,2),'g', 'LineWidth', 5)
hold on; 
plot(PositiveExamples{1}(:,1),PositiveExamples{1}(:,2),'r*')
hold on; 
plot(NegativeExamples{1}(:,1),NegativeExamples{1}(:,2),'b*')
%}
initialCaseIndex    = ones(numberExamplesPerTaskForTraining,1);
CaseIndexes         = zeros(numberExamplesPerTaskForTraining,1);
for i=1:1:Number_Of_Tasks-2
    CaseIndexes = [ CaseIndexes; i*initialCaseIndex ];
end

rotation_angles = calculateRotationAngles(-0.75*pi,-0.25.*pi, Number_Of_Tasks);%0.4
TensorOfTestData = cell(Number_Of_Tasks,1);
TensorOfRotatedPositiveExamples = cell(Number_Of_Tasks,1);
TensorOfRotatedNegativeExamples  = cell(Number_Of_Tasks,1);
TensorOfRotatedElipse = cell(Number_Of_Tasks,1);
TrainingLabelsPerTask = cell(Number_Of_Tasks,1);
TensorOfLabelsForTestData = cell(Number_Of_Tasks,1); 
for taskCounter = 1:1:Number_Of_Tasks
    RotatedPositiveExamples = rotateMatrix( PositiveExamples{taskCounter},rotation_angles(taskCounter));
    RotatedNegativeExamples = rotateMatrix( NegativeExamples{taskCounter},rotation_angles(taskCounter));
    RotatedElipse = rotateMatrix(Elipse, rotation_angles(taskCounter)); 
    TensorOfRotatedPositiveExamples{taskCounter} = RotatedPositiveExamples;
    TensorOfRotatedNegativeExamples{taskCounter} = RotatedNegativeExamples;
    TensorOfRotatedElipse{taskCounter} = RotatedElipse;
    
    for exampleCounter = 1:1:numberOfPositiveExamplesForTraining
        TensorOfTrainingDataPerTask_Matrix(taskCounter,exampleCounter,1) = RotatedPositiveExamples(exampleCounter , 1 );
        TensorOfTrainingDataPerTask_Matrix(taskCounter,exampleCounter,2) = RotatedPositiveExamples(exampleCounter , 2 ); 
    end
    
    for exampleCounter = 1:1:numberOfNegativeExamplesForTraining
        TensorOfTrainingDataPerTask_Matrix(taskCounter,numberOfPositiveExamplesForTraining+exampleCounter, 1) = RotatedNegativeExamples(exampleCounter,1);
        TensorOfTrainingDataPerTask_Matrix(taskCounter,numberOfPositiveExamplesForTraining+exampleCounter, 2) = RotatedNegativeExamples(exampleCounter,2);   
    end
    TrainingLabelsPerTask{taskCounter} = [zeros( numberOfPositiveExamplesForTraining, 1); ones(numberOfNegativeExamplesForTraining,1) ];
    
    TensorOfTestData{taskCounter} = [ RotatedPositiveExamples(numberOfPositiveExamplesForTraining+1: length(RotatedPositiveExamples), :) ; RotatedNegativeExamples(numberOfNegativeExamplesForTraining+1:length(RotatedNegativeExamples) , : ) ];
    TensorOfLabelsForTestData{taskCounter} = [zeros( length(RotatedPositiveExamples)- numberOfPositiveExamplesForTraining, 1); ones(length(RotatedNegativeExamples) - numberOfNegativeExamplesForTraining,1)];
    
    clear RotationMatrix;
    clear RotatedPositiveExamples;
    clear RotatedNegativeExamples;
    clear RotatedElipse;
end

    TensorOfTrainingDataPerTask = cell(Number_Of_Tasks,1);
for taskCounter = 1:1:Number_Of_Tasks
    TensorOfTrainingDataPerTask{taskCounter} = TensorOfTrainingDataPerTask_Matrix(taskCounter,:,:);
end

% Draw a new figure and set the default values for the text font
fig = figure;
set(fig,'DefaultAxesFontName', 'Arial')
set(fig,'DefaultAxesFontSize', 16)
% Let's plot the data
for numberOfTasks = 1:1:Number_Of_Tasks
    h = plot(TensorOfRotatedElipse{numberOfTasks}(:,1),TensorOfRotatedElipse{numberOfTasks}(:,2),'g', 'LineWidth', 0.4); 
    axis([-1.1 1.1  -1.1 1.1]);
    hold on; 
    h = plot(TensorOfRotatedPositiveExamples{numberOfTasks}(:,1),TensorOfRotatedPositiveExamples{numberOfTasks}(:,2),'r*');
    axis([-1.1 1.1  -1.1 1.1]);
    hold on;
    h= plot(TensorOfRotatedNegativeExamples{numberOfTasks}(:,1),TensorOfRotatedNegativeExamples{numberOfTasks}(:,2),'b*');
end
plot2svg('Task.svg')




checkForFirstTime = 1;

TensorOfTrainingData = cell(Number_Of_Tasks,1);
TensorOfLabelsForTrainingData = cell(Number_Of_Tasks,1);
for testTaskCounter = 1:1:Number_Of_Tasks
    for trainingTaskCounter = 1:1:Number_Of_Tasks
        if testTaskCounter ~= trainingTaskCounter
            tmp1 =  TensorOfTrainingData{testTaskCounter};
            tmp2 =  TensorOfTrainingDataPerTask{trainingTaskCounter};
            TensorOfTrainingData{testTaskCounter} = [tmp1,  tmp2];
            
            lbl1 =  TensorOfLabelsForTrainingData{testTaskCounter};
            lbl2 = TrainingLabelsPerTask{trainingTaskCounter};
            
            TensorOfLabelsForTrainingData{testTaskCounter} = [lbl1;  lbl2];
        end
    end
end

%{
for taskCounter=1:1:Number_Of_Tasks
    TrainingFileName = strcat('TrainingData_For_Case_',num2str(taskCounter-1),'_numberOfTasks_',num2str(Number_Of_Tasks),'_numberOfTotalTrainingPoints_',num2str(Number_Of_Tasks*numberExamplesPerTaskForTraining),'_numberOfTrainingPointsPerEachTask_',num2str(numberExamplesPerTaskForTraining),'_','Repeat_Number_',num2str(repeat_Number),'_','Run_',num2str(Run),'.csv');
    TestFileName = strcat('TestData_For_Case_',num2str(taskCounter-1),'_numberOfTasks_',num2str(Number_Of_Tasks),'_numberOfTotalTrainingPoints_',num2str(Number_Of_Tasks*numberExamplesPerTaskForTraining),'_numberOfTrainingPointsPerEachTask_',num2str(numberExamplesPerTaskForTraining),'_','Repeat_Number_',num2str(repeat_Number),'_','Run_',num2str(Run),'.csv');
    LabelsTrainingFileName = strcat('LabelsFor_TrainingData_For_Case_',num2str(taskCounter-1),'_numberOfTasks_',num2str(Number_Of_Tasks),'_numberOfTotalTrainingPoints_',num2str(Number_Of_Tasks*numberExamplesPerTaskForTraining),'_numberOfTrainingPointsPerEachTask_',num2str(numberExamplesPerTaskForTraining),'_','Repeat_Number_',num2str(repeat_Number),'_','Run_',num2str(Run), '.csv');
    LabelsTestFileName = strcat('LabelsFor_TestData_For_Case_',num2str(taskCounter-1),'_numberOfTasks_',num2str(Number_Of_Tasks),'_numberOfTotalTrainingPoints_',num2str(Number_Of_Tasks*numberExamplesPerTaskForTraining),'_numberOfTrainingPointsPerEachTask_',num2str(numberExamplesPerTaskForTraining),'_','Repeat_Number_',num2str(repeat_Number),'_','Run_',num2str(Run), '.csv');
    CaseIndexesFileName = strcat('CaseIndexes_For_Case_',num2str(taskCounter-1),'_numberOfTasks_',num2str(Number_Of_Tasks),'_numberOfTotalTrainingPoints_',num2str(Number_Of_Tasks*numberExamplesPerTaskForTraining),'_numberOfTrainingPointsPerEachTask_',num2str(numberExamplesPerTaskForTraining),'_','Repeat_Number_',num2str(repeat_Number),'_','Run_',num2str(Run), '.csv');
    
    convertedTrainingData =  TensorOfTrainingData{taskCounter};
    sizeOfTrainingData = size(convertedTrainingData);
    convertedTrainingDataReshaped = reshape(convertedTrainingData, sizeOfTrainingData(2), 2);
    shuffledIndexes = randperm(sizeOfTrainingData(2))';
    
    convertedTestData =  TensorOfTestData{taskCounter};
    
    Shuffled_TrainingData = convertedTrainingDataReshaped( shuffledIndexes , : );
    tmpTrainingLabels = TensorOfLabelsForTrainingData{taskCounter};
    Shuffled_TrainingLabels = tmpTrainingLabels( shuffledIndexes , : );
    Shuffled_CaseIndexes = CaseIndexes( shuffledIndexes , : );
    
    dlmwrite(TrainingFileName, Shuffled_TrainingData, 'precision', '%.6f','newline','pc');   
    dlmwrite(LabelsTrainingFileName, Shuffled_TrainingLabels, 'precision', '%.6f','newline','pc'); 
    dlmwrite(CaseIndexesFileName, Shuffled_CaseIndexes, 'precision', '%.6f','newline','pc'); 
   
    dlmwrite(TestFileName, convertedTestData, 'precision', '%.6f','newline','pc'); 
    tmpTestLabels = TensorOfLabelsForTestData{taskCounter};
    dlmwrite(LabelsTestFileName, tmpTestLabels, 'precision', '%.6f','newline','pc');

    
end

%}
