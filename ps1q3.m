clc
clear
close all

%% params
alpha = 1/3;
r = 0.05;
delta = 0.01;
g = 0;
w = 1;
mu = 0;
rho = 0.95;
sigma = 0.01;
m = 3;

%% grids
Np      = 20;   % size of grid for p
Nk      = 300;  % size of grid for K

% capital grid
l_kss = ((1-alpha)/(1-2*alpha))*((1/(1-alpha))*(log(alpha)-alpha*(log(1+g)+log(w)))-log(1+r)-log(1-(1-delta)/(1+r)*exp(mu)));
kss=exp(l_kss);
k_min=kss/3;
k_max=3*kss;
k_grid = linspace(k_min,k_max,Nk)';
k_mat = repmat(k_grid,[1,Np,Nk]);
kp_grid(1,1,:) = k_grid;
kp_mat = repmat(kp_grid,[Nk,Np,1]);

% do the tauchen stuff
[logp_grid,p_ij] = tauchen(Np,mu,rho,sigma,m);

%% helpers
p_grid = exp(logp_grid)';
p_mat = repmat(p_grid,[Nk,1,Nk]);
pi_mat    = (1-alpha).*(alpha*k_mat/w).^(alpha/(1-alpha));
inv_cost_mat = p_mat .* (kp_mat - (1-delta)*k_mat);

%% vfi 1

V0=zeros(Nk,Np);

diff=10;
while (diff>1e-6)
    continuation_value=permute(repmat(p_ij*V0',[1,1,Nk]),[3,1,2]);
    flow_value = pi_mat-inv_cost_mat;
    V=max(flow_value+continuation_value/(1+r),[],3);
    diff=max(max(abs(V0-V),[],1),[],2);
    V0=V;
end

% Get the policy function
[V_stochastic,pol_stochastic]=max(flow_value+continuation_value/(1+r),[],3);
max_diff = max(abs(max(k_grid(pol_stochastic),[],1)-min(k_grid(pol_stochastic),[],1)));

%% graph
analytical_kp = 1/(1-2*alpha)*(log(alpha)-log(w)-(1-alpha)*( log(1+r)+log(p_grid)+log(1-p_grid.^(rho-1)*(1-delta)/(1+r)*exp(sigma^2/2+mu))));

kp = zeros(1,Np);
for i = 1:Np
    kp(i) = log(k_grid(pol_stochastic(1,i)));
end

h1 = figure(1);
plot(logp_grid,kp,'-')
hold on
plot(logp_grid,analytical_kp,'--')
hold off
legend('numerical','analytical')
xlabel('log(p^K)')
ylabel('log(K)')

%%  vfi deterministic
sigma= 0;
p_ij = zeros(length(logp_grid));
for i=1:Np
    p_next = p_grid(i)^rho;
    j = find(p_grid > p_next,1);
    coef = (p_next-p_grid(j-1))/(p_grid(j)-p_grid(j-1));
    p_ij(i,j) = coef;
    p_ij(i,j-1) = 1-coef;
end

V0=zeros(Nk,Np);

diff=10;
while (diff>1e-6)
    continuation_value=permute(repmat(p_ij*V0',[1,1,Nk]),[3,1,2]);
    flow_value = pi_mat-inv_cost_mat;
    V=max(flow_value+continuation_value/(1+r),[],3);
    diff=max(max(abs(V0-V),[],1),[],2);
    V0=V;
end
% Get the policy function
[V_deterministic,pol_deterministic]=max(flow_value+continuation_value/(1+r),[],3);

kp_deterministic = zeros(1,Np);
for i = 1:Np
    kp_deterministic(i) = log(k_grid(pol_deterministic(1,i)));
end


%% graphing

h2 = figure(2);
plot(logp_grid,kp,'-')
hold on
plot(logp_grid,kp_deterministic,'--')
hold off
legend('stochastic','deterministic')
xlabel('log(p^K)')
ylabel('log(K)')
