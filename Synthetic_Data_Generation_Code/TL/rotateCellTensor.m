function RotatedMatrix = rotateCellTensor(matrix, rotation_angle)
converted_matrix = cell2mat(matrix);
RotatedMatrix = zeros(size(converted_matrix,1), size(converted_matrix,2));

RotationMatrix= [ cos(rotation_angle)  -sin(rotation_angle); sin(rotation_angle) cos(rotation_angle)];

for i = 1:1:size(converted_matrix,1)
        RotatedMatrix(i, : ) = converted_matrix(i, :) * RotationMatrix;
end