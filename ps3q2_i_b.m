clc
clear
close all

%% params
theta = 1/3;
beta = 0.9;
R = 0.04;
delta = 0;
mu = 0;
rho = 0.85;
sigma = 0.05;
m = 3;
F = 0.03;
P = 0.02;


%% grids
Nz      = 12;   % size of grid for p
Nk      = 300;  % size of grid for K

% capital grid
k_min=0.1;
k_max=90;
k_grid = linspace(k_min,k_max,Nk)';
k_mat = repmat(k_grid,[1,Nz,Nk]);

% do the tauchen stuff
[logz_grid,p_ij] = tauchen(Nz,mu,rho,sigma,m);

%% helpers
z_grid = exp(logz_grid)';
z_mat = repmat(z_grid,[Nk,1,Nk]);
pi_mat = (z_mat) .* (k_mat.^theta) - (k_mat * R);
kp_mat = permute(k_mat, [3,2,1]);
[errors, depreciated_k] = min(abs(repmat(k_grid,[1,Nk])*(1-delta)-repmat(k_grid',[Nk,1])),[],2); 

%% vfi 1

V0=zeros(Nk,Nz);
V0_adj = zeros(Nk,Nz);
V0_noadj = zeros(Nk,Nz);

diff=10;
iters = 0;
while (diff>1e-6)
    EV_cont = V0*p_ij';
    EV_cont_noadj = EV_cont(depreciated_k,:);
    EV_cont_adj=permute(repmat(EV_cont,[1 1 Nk]),[3,2,1]);
    [V_adj, pol_adj] = max(pi_mat*(1-P) + beta*EV_cont_adj,[],3); 
    V_noadj = pi_mat(:,:,1) + beta*EV_cont_noadj;
    V = max(V_noadj,V_adj);
    diff_adj = max(max(abs(V0_adj-V_adj),[],1),[],2);
    diff_noadj = max(max(abs(V0_noadj-V_noadj),[],1),[],2);
    diff = max(diff_adj, diff_noadj);
    iters = iters+1;
    V0 = V; 
    V0_noadj = V_noadj; 
    V0_adj = V_adj;
end

% determine bands
should_adjust = V_adj > V_noadj;
k_star = k_grid(pol_adj(1,:));
offset_k_mat = k_mat(2:Nk,:);
upper_band = zeros(1,Nz);
lower_band = zeros(1,Nz);
upper_band(1,:) = offset_k_mat(should_adjust(2:Nk,:) & ~should_adjust(1:Nk-1,:));
lower_band(1,:) = offset_k_mat(~should_adjust(2:Nk,:) & should_adjust(1:Nk-1,:));


%% graph

h1 = figure(1);
plot(logz_grid,upper_band,'-')
hold on
plot(logz_grid,lower_band,'-')
hold on
plot(logz_grid,k_star,'-')
hold off
legend('upper band','lower band','k^*')
xlabel('log(z)')
ylabel('K')
ylim([0,90])

