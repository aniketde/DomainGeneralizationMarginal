
counter = 1;
for u = 0:0.01:1
    y(counter) = 1 / (1 + exp(-u));
    counter = counter + 1;
end


plot(y);