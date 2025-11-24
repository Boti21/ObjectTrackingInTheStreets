%% Checking Observabilty of State-Space Models
dt = 1;   

% State transition matrix F
F = [1 0 0 0 dt 0;
     0 1 0 0 0  dt;
     0 0 1 0 0  0;
     0 0 0 1 0  0;
     0 0 0 0 1  0;
     0 0 0 0 0  1];

% Observation matrix H
H = [1 0 0 0 0 0;
     0 1 0 0 0 0;
     0 0 1 0 0 0;
     0 0 0 1 0 0];

O = obsv(F,H);

if size(F,1) == rank(O)
    disp("System is observable!")
else
    disp("System is NOT observable!")
end

