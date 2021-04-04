clc
clear
close all

%% params
global r lambda c z alpha p_grid p_ij
r = 0.01;
lambda = 0.01;
alpha = 0.33;
rho = 0.95;
mu = 0;
sigma = 0.02;
z = 0.7;
c = 0.1;
m = 3;

%% grids
N      = 11;   % size of grid for p

% tauchen
[logp_grid,p_ij] = tauchen(N,mu,rho,sigma,m);
p_grid = exp(logp_grid);


%% fsolve

theta_val = fmincon(@(x) diff_equation(x), zeros(N,1));

%% graphing

h1 = figure(1);
plot(p_grid,theta_val,'-')
xlabel('p')
ylabel('theta(p)')

%% sim
T = 1000;
rng(123456);

%draw a random vector which determines when to change state
random_numbers=rand(T,1);
has_shock=(1/3>random_numbers);
states=zeros(T,1);
states(1)= (N + 1) / 2; 

for i=2:T
    if has_shock(i)==0
        states(i)=states(i-1);
    else
       discrete_cdf = p_ij(states(i-1),:)*triu(ones(N,N));
       states(i)=sum(rand() > discrete_cdf) + 1;
    end
    
end

theta_path = theta_val(states);
logtheta_path = log(theta_path);
logp_path = logp_grid(states);
ratio = std(logtheta_path)/std(logp_path)


%% function definitions

function [error] = diff_equation(theta) 

global r lambda c z alpha p_grid p_ij

% theta needs to be N x 1

lhs = (2*(r + lambda)*sqrt(theta) + theta) * c;
rhs = p_grid - z + 2 * c * alpha * (p_ij*sqrt(theta) - sqrt(theta));
diff = lhs - rhs;
error = diff'*diff;

end