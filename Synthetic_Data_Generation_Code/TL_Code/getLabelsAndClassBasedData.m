function [PositiveExamples,NegativeExamples] = getLabelsAndClassBasedData(TensorOfTasks,NumberOfPointsPerTask, Number_Of_Tasks)
for taskCounter = 1:1:Number_Of_Tasks
    positiveExampleCounter = 1;
    negativeExampleCounter = 1;
    for pointCounter = 1:1:NumberOfPointsPerTask
        if(TensorOfTasks(taskCounter,pointCounter,2)>0)
            PositiveExamples_tmp(positiveExampleCounter, : ) = TensorOfTasks(taskCounter,pointCounter, :);
            positiveExampleCounter = positiveExampleCounter + 1;
        else
            NegativeExamples_tmp(negativeExampleCounter, : ) = TensorOfTasks(taskCounter,pointCounter, :);
            negativeExampleCounter = negativeExampleCounter + 1;
        end
    end
    PositiveExamples{taskCounter} = PositiveExamples_tmp;
    NegativeExamples{taskCounter} = NegativeExamples_tmp;
    clear PositiveExamples_tmp;
    clear NegativeExamples_tmp;
end
