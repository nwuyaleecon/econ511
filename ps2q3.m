clc
clear
close all

%% params
a = 2;
omega = 0.5;
beta = 0.99;
p = 1;

%% grids
Nk      = 2000;  % size of grid for K

% capital grid
k_min=1;
k_max=4;
k_grid = linspace(k_min,k_max,Nk)';
k_mat = repmat(k_grid,[1,Nk]);
kp_mat = repmat(k_grid',[Nk,1]);
i_mat = kp_mat - k_mat;
c_mat = zeros(Nk, Nk);
c_mat(i_mat > 0) = i_mat(i_mat > 0).^2;
c_mat(i_mat < 0) = -1*omega*i_mat(i_mat < 0);
flow_value = a*k_mat - (k_mat.^2)/2 - p*(i_mat + c_mat);

%% vfi 

V0=zeros(Nk,1);

diff=10;
while (diff>1e-6)
    
    V=max(flow_value + beta*repmat(V0',[Nk,1]),[],2);
    diff=max(abs(V0-V));
    V0=V;
end

% Get the policy function
[V,pol]=max(flow_value + beta*repmat(V0',[Nk,1]),[],2);
i_pol = k_grid(pol) - k_grid;

%% graph

h1 = figure(1);
plot(k_grid,V,'-')
xlabel('K')
ylabel('V(K)')

dV_dK = (V(2:end) - V(1:(end-1))) ./ (k_grid(2:end) - k_grid(1:(end-1)));

h2 = figure(2);
plot(dV_dK, i_pol(1:(end-1)))
xlabel("V'");
ylabel("I(V')");


