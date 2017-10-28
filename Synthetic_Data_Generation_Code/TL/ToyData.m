clear all
clc
cla

NumberOfPoints = 25;

M(1,1) = rand(1);
M(1,2) = (1-M(1,1))*rand(1);
factor = 5;
elementCounter = 2;
sizeOfM = size(M);
while (sizeOfM(1) < NumberOfPoints)
    x = rand(1);
    x = -1+2*x;
    y_bound = 1 - x^2;
    y_bound = sqrt(y_bound);
    y = -1*y_bound  + 2*y_bound*rand(1);
    
    r = x^2 + y^2;
    if r <= 1
       M(elementCounter, 1) = x;
       M(elementCounter, 2) = factor*y;
       elementCounter = elementCounter +1;
    end
    sizeOfM = size(M);    
end

plot(M(:,1),M(:,2),'r*')


Labels = zeros(NumberOfPoints,1);
positiveExampleCounter = 1;
negativeExampleCounter = 1;
for i=1:NumberOfPoints
    if M(i,1) >= 0
        Labels(i) = 1;
        PositiveExamples(positiveExampleCounter, : ) = M(i, : );
        positiveExampleCounter = positiveExampleCounter + 1;
    else
        NegativeExamples(negativeExampleCounter, : ) = M(i, : );
        negativeExampleCounter = negativeExampleCounter + 1;
    end
end

Elipse = calculateEllipse(0, 0, 1, 5, 0, 10000);
%Elipse = Elipse';
plot(Elipse(: , 1),Elipse(:,2),'g', 'LineWidth', 5); hold on; plot(PositiveExamples(:,1),PositiveExamples(:,2),'r*');hold on; plot(NegativeExamples(:,1),NegativeExamples(:,2),'b*')
%plot(PositiveExamples(:,1),PositiveExamples(:,2),'r*');hold on; plot(NegativeExamples(:,1),NegativeExamples(:,2),'b*')


angleCounter = 1;
for rotation_angle = -1.0*pi/4:pi/12:pi/4
    RotationMatrix= [ cos(rotation_angle)  -sin(rotation_angle); sin(rotation_angle) cos(rotation_angle)];
    RotatedPositiveExamples = zeros(length(PositiveExamples) ,2);
    for i = 1:1:length(PositiveExamples )
        RotatedPositiveExamples(i, : ) = PositiveExamples(i, :) * RotationMatrix;
    end

    RotatedNegativeExamples = zeros(length(NegativeExamples) ,2);
    for i = 1:1:length(NegativeExamples )
        RotatedNegativeExamples(i, : ) = NegativeExamples(i, :) * RotationMatrix;
    end

    RotatedElipse = zeros(length(Elipse) ,2);
    for i = 1:1:length(Elipse )
        RotatedElipse(i, : ) = Elipse(i, :) * RotationMatrix;
    end

    
    %plot(RotatedElipse(:,1),RotatedElipse(:,2)); hold on; plot(RotatedPositiveExamples(:,1),RotatedPositiveExamples(:,2),'r*');hold on; plot(RotatedNegativeExamples(:,1),RotatedNegativeExamples(:,2),'b*')
    %pbaspect([4*r1 r2 1 ]);
    TensorOfRotationMatrix(angleCounter,:,:) = RotationMatrix;
    TensorOfRotatedPositiveExamples(angleCounter,:,:) = RotatedPositiveExamples;
    TensorOfRotatedNegativeExamples(angleCounter,:,:) = RotatedNegativeExamples;
    TensorOfRotatedElipse(angleCounter,:,:) = RotatedElipse;

    angleCounter = angleCounter + 1;
    clear RotationMatrix;
    clear RotatedPositiveExamples;
    clear RotatedNegativeExamples;
    clear RotatedElipse;
end

for numberOfTasks = 1:1:angleCounter-1
    plot(TensorOfRotatedElipse(numberOfTasks,:,1),TensorOfRotatedElipse(numberOfTasks,:,2),'g', 'LineWidth', 5); hold on; plot(TensorOfRotatedPositiveExamples(numberOfTasks,:,1),TensorOfRotatedPositiveExamples(numberOfTasks,:,2),'r*');hold on; plot(TensorOfRotatedNegativeExamples(numberOfTasks,:,1),TensorOfRotatedNegativeExamples(numberOfTasks,:,2),'b*')
end

numberOfPositiveExamples                = length(TensorOfRotatedPositiveExamples(1,:,:));
numberOfNegativeExamples                = length(TensorOfRotatedNegativeExamples(1,:,:));
numberOfPositiveExamplesForTraining     = floor(0.7*numberOfPositiveExamples); 
numberOfNegativeExamplesForTest         = floor(0.7*numberOfNegativeExamples);

for numberOfTasks = 1:1:angleCounter-1
    TensorOfTrainingData(numberOfTasks,:,:) = [TensorOfRotatedPositiveExamples(numberOfTasks,1:numberOfPositiveExamplesForTraining,:), TensorOfRotatedNegativeExamples(numberOfTasks,1:numberOfNegativeExamplesForTest,:)];
    TensorOfTestData(numberOfTasks,:,:) = [TensorOfRotatedPositiveExamples(numberOfTasks,numberOfPositiveExamplesForTraining+1:numberOfPositiveExamples, :), TensorOfRotatedNegativeExamples(numberOfTasks,numberOfNegativeExamplesForTest+1:numberOfNegativeExamples, :)];
end

TrainingLabels = [ones(numberOfPositiveExamplesForTraining,1); 0*ones(numberOfNegativeExamplesForTest,1)];
TestLabels = [ones( numberOfPositiveExamples - numberOfPositiveExamplesForTraining,1); 0*ones(numberOfNegativeExamples - numberOfNegativeExamplesForTest,1)];

TrainingLabels = [TrainingLabels; TrainingLabels; TrainingLabels; TrainingLabels; TrainingLabels; TrainingLabels];


initialCaseIndex = ones(numberOfPositiveExamplesForTraining+numberOfNegativeExamplesForTest,1);
CaseIndexes = [ 0*initialCaseIndex; initialCaseIndex; 2*initialCaseIndex; 3*initialCaseIndex; 4*initialCaseIndex; 5*initialCaseIndex ];

length(CaseIndexes)


%{
Training = 'TrainingData_For_Case_';
Test = 'TestData_For_Case_';
LabelsForTraining = 'Labels_For_TrainingData_For_Case_';
LabelsForTest = 'Labels_For_TestData_For_Case_';
CaseIndexFile = 'CaseIndices_For_TrainingData_For_Case_';




for numberOfTasks = 1:1:angleCounter-1   
    taskCounterForTrainingData = 1;
    TrainingDataFileName = strcat( Training, num2str(numberOfTasks-1), '.csv');
    TestDataFileName = strcat( Test, num2str(numberOfTasks-1), '.csv');
    TrainingLabelsFileName = strcat(LabelsForTraining,  num2str(numberOfTasks-1),'.csv');
    TestLabelsFileName = strcat(LabelsForTest, num2str(numberOfTasks-1), '.csv');
    CaseIndexFileName = strcat(CaseIndexFile, num2str(numberOfTasks-1), '.csv');
    sizeOfTrainingData = size(TensorOfTrainingData(1,:,:));
    sizeOfTestData = size(TensorOfTestData(1,:,:));
    a = zeros( (angleCounter-2)*sizeOfTrainingData(2), sizeOfTrainingData(3) );
    
    for writeCounter = 1:1:angleCounter-1
        if writeCounter ~= numberOfTasks
            temp = reshape(TensorOfTrainingData(writeCounter,:,:), [sizeOfTrainingData(2) sizeOfTrainingData(3)]);
            for exampleCounter = 1:1:length(temp)
                a( (taskCounterForTrainingData -1)*sizeOfTrainingData(2) + exampleCounter , :) = temp(exampleCounter , :);
            end
            taskCounterForTrainingData = taskCounterForTrainingData +1;
        else
            b = reshape(TensorOfTestData(writeCounter,:,:), [sizeOfTestData(2) sizeOfTestData(3)]);
            dlmwrite(TestDataFileName, b, 'precision', '%.6f','newline','pc');
            dlmwrite(TestLabelsFileName, TestLabels, 'precision', '%.0f','newline','pc');
        end
        
            shuffledIndexes = randperm(length(a))';
            Shuffled_a = a( shuffledIndexes , : );
            Shuffled_TrainingLabels = TrainingLabels( shuffledIndexes , : );
            Shuffled_CaseIndexes = CaseIndexes( shuffledIndexes , : );
            dlmwrite(TrainingDataFileName, Shuffled_a, 'precision', '%.6f','newline','pc');
            dlmwrite(TrainingLabelsFileName, Shuffled_TrainingLabels, 'precision', '%.0f','newline','pc');
            dlmwrite(CaseIndexFileName, Shuffled_CaseIndexes,'precision', '%.0f','newline','pc');
    end
    
end

%}
