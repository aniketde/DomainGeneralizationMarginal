function [TensorOfTasks] = getTaskData(NumberOfPointsPerTask, Number_Of_Tasks, factor)

for taskCounter = 1:1:Number_Of_Tasks
    TensorOfTasks(taskCounter,1,1) = 0;
    TensorOfTasks(taskCounter,1,2) = 0;
end

for taskCounter = 1:1:Number_Of_Tasks
    elementCounter = 2;
    while(elementCounter <= NumberOfPointsPerTask)  
        x = rand(1);
        x = -1+2*x;
        y_bound = 1 - x^2;
        y_bound = sqrt(y_bound);
        y = -1*y_bound  + 2*y_bound*rand(1);
        r = x^2 + y^2; 
        if r<=1 %&& x <= 0.0
           %{ 
            p = rand;
            if p >= 0.5
                TensorOfTasks(taskCounter, elementCounter, 1) = x+rand*0.1;
                TensorOfTasks(taskCounter, elementCounter, 2) = factor*(y+rand*0.1);
                elementCounter = elementCounter +1;
            else
                TensorOfTasks(taskCounter, elementCounter, 1) = x-rand*0.1;
                TensorOfTasks(taskCounter, elementCounter, 2) = factor*(y-rand*0.1);
                elementCounter = elementCounter +1;        
            end
            %}
            TensorOfTasks(taskCounter, elementCounter, 1) = x;
            TensorOfTasks(taskCounter, elementCounter, 2) = factor*y;
            elementCounter = elementCounter +1;
            
        end
    end
end

%plot(TensorOfTasks(1,:,1),TensorOfTasks(1,:,2),'r*')
%hold on
%plot(TensorOfTasks(2,:,1),TensorOfTasks(2,:,2),'g*')



