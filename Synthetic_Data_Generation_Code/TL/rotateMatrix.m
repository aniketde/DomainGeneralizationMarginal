function RotatedMatrix = rotateMatrix(matrix, rotation_angle)
RotatedMatrix = zeros(size(matrix,1), size(matrix,2));

RotationMatrix= [ cos(rotation_angle)  -sin(rotation_angle); sin(rotation_angle) cos(rotation_angle)];

for i = 1:1:size(matrix,1)
        RotatedMatrix(i, : ) = matrix(i, :) * RotationMatrix;
end