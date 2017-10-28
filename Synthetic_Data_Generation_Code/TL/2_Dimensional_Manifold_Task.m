
phi = pi*0.2;
a = 4;
b = 2;
X_c = 0;
Y_c = 0;
counter = 1;

t = 0:.01:2*pi;
X = X_c + a*cos(t)*cos(phi) - b*sin(t)*sin(phi);
Y = Y_c + a*cos(t)*sin(phi) - b*sin(t)*cos(phi);
plot(X,Y)