function rotation_angles = calculateRotationAngles(startAngle, endAngle,Number_Of_Tasks)

%angleIncrement = (startAngle - endAngle )/(Number_Of_Tasks +1);

%for i = 1:1:Number_Of_Tasks
%    rotation_angles(i) = (startAngle + i * angleIncrement);
%end
%{
totalAngle  = (startAngle - endAngle );

angleIncrement = totalAngle/(Number_Of_Tasks-1);

rotation_angles(1) = startAngle
for i = 2:1:Number_Of_Tasks - 1; 
    rotation_angles(i) = ( rotation_angles(i-1) + abs((0.95*angleIncrement) + (1-0.95).*rand()*angleIncrement));
end
rotation_angles(Number_Of_Tasks) = endAngle;
%}

 rotation_angles = unifrnd(startAngle,endAngle,1,Number_Of_Tasks);