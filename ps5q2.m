clc
clear
close all

global beta min_B eta alpha p A L delta N K shock_mat T
beta = 0.96;
min_B = 0;
delta = 0.1;
alpha = 0.36;
p = 0.5;
A = 1;
L = 0.5;
N = 250;
% initialize grid of capital
K_min = min_B;
K_max = 10;
K = linspace(K_min, K_max, N);
K_edges = linspace(K_min, K_max, 50);
C_edges = linspace(0, 2, 25);
L_star = L;
T = 10000;
shock_mat = rand(T, N);
dummy_function = @(x) (iterate_K(x)-x)^2;

% low risk
eta = 0.2;
K_star1 = fminbnd(dummy_function,2,3);
[w_star1, r_star1] = get_equilibrium_values(K_star1);
display([K_star1, L_star, w_star1, r_star1])
[~, pol_H1, pol_L1, V_H1, V_L1] = iterate_K(K_star1);
[c_pol_H1, c_pol_L1] = get_consumption_policy(pol_H1, pol_L1, w_star1, r_star1);
[k_path_1, c_path_1, w_path_1] = get_simulated_paths(pol_H1, pol_L1, r_star1, w_star1);

% high risk 
eta = 0.01;
K_star2 = fminbnd(dummy_function,2,3);
[w_star2, r_star2] = get_equilibrium_values(K_star2);
display([K_star2, L_star, w_star2, r_star2])
[~, pol_H2, pol_L2, V_H2, V_L2] = iterate_K(K_star2);
[c_pol_H2, c_pol_L2] = get_consumption_policy(pol_H2, pol_L2, w_star2, r_star2);
[k_path_2, c_path_2, w_path_2] = get_simulated_paths(pol_H2, pol_L2, r_star2, w_star2);

% plotting
display([mean(mean(w_path_1)), mean(mean(w_path_2))])

figure(1)
subplot(2,2,1)
histogram(K(k_path_1), K_edges, 'Normalization','probability');
hold on
histogram(K(k_path_2), K_edges, 'Normalization','probability');
title('Distribution of Assets')
legend('eta=0.2','eta=0.01')
subplot(2,2,2)
histogram(c_path_1, C_edges, 'Normalization','probability');
hold on
histogram(c_path_2, C_edges, 'Normalization','probability');
title('Distribution of Consumption')
legend('eta=0.2','eta=0.01')
subplot(2,2,3)
plot(K, c_pol_L1, K, c_pol_H1, K, c_pol_L2, K, c_pol_H2)
title('Consumption Policy')
legend('eta=0.2, low state','eta=0.2, high state','eta=0.01, low state','eta=0.01, high state')
subplot(2,2,4)
plot(K, [V_L1, V_H1, V_L2, V_H2])
title('Value Function')
legend('eta=0.2, low state','eta=0.2, high state','eta=0.01, low state','eta=0.01, high state')


function [k_path, c_path, welfare_path] = get_simulated_paths(policy_H, policy_L, rate, wage)

global shock_mat T N K eta

k_path = zeros(T,N);
c_path = zeros(T-1,N);
welfare_path = zeros(T-1,N);
k_path(1,:) = 1:N;

for k=1:N
    t = 2;
    while t < T+1
        if shock_mat(t,k) > 0.5
            k_path(t,k) = policy_H(k_path(t-1,k));
            c_path(t-1,k) = K(k_path(t-1,k)) * (1+rate) - K(k_path(t,k)) + (1-eta)*wage;
        else
            k_path(t,k) = policy_L(k_path(t-1,k));
            c_path(t-1,k) = K(k_path(t-1,k)) * (1+rate) - K(k_path(t,k)) + (eta)*wage;
        end
        welfare_path(t-1,k) = log(c_path(t-1,k));
        t = t + 1;
    end
end

end

function [c_pol_H, c_pol_L] = get_consumption_policy(pol_H, pol_L, wage, rate)

global K N eta

c_pol_H = zeros(N,1);
c_pol_L = zeros(N,1);
for i = 1:N
    c_pol_H(i) =  K(i)*(1+rate) - K(pol_H(i)) + (1-eta)*wage;
    c_pol_L(i) =  K(i)*(1+rate) - K(pol_L(i)) + (eta)*wage;
end
    
end

function [wage, rental_rate] = get_equilibrium_values(K_guess)

global alpha L delta A

wage = (1-alpha)*A*(K_guess^alpha)*(L^(-1*alpha));
rental_rate = alpha*A*(K_guess^(alpha - 1))*(L^(1 - alpha)) - delta;

end


function [Khat, pol_H, pol_L, V_H, V_L] = iterate_K(K_guess) 

global N K eta p beta

[w, r] = get_equilibrium_values(K_guess);

% precompute U at each pair k, k'
U_H = zeros(N,N);
U_L = zeros(N,N);
for i = 1:N
    for j = 1:N
        k = K(i);
        kp = K(j);
        U_L(i,j) = log(max(eta.*w + (1+r)*k - kp, 1e-100));
        U_H(i,j) = log(max((1-eta).*w + (1+r)*k - kp, 1e-100));
    end
end

V_H = zeros(N,1);
V_L = zeros(N,1);
error = 1;
tolerance = 1e-6;
while error > tolerance
    V_H_new = max(U_H + beta*p*repmat(V_H',N,1) + beta*(1-p)*repmat(V_L',N,1), [], 2);
    V_L_new = max(U_L + beta*p*repmat(V_L',N,1) + beta*(1-p)*repmat(V_H',N,1), [], 2);
    
    error = max(max(abs(V_L_new - V_L), [],'all'), max(abs(V_H_new - V_H), [], 'all'));
    V_L = V_L_new;
    V_H = V_H_new;
end

% extract policy functions
[~, pol_H] = max(U_H + beta*p*repmat(V_H',N,1) + beta*(1-p)*repmat(V_L',N,1), [], 2);
[~, pol_L] = max(U_L + beta*p*repmat(V_L',N,1) + beta*(1-p)*repmat(V_H',N,1), [], 2);

% use eigenvector method on transition matrix
% treat states 1-N as high states, N+1-2N as low states
M = zeros(2*N, 2*N);
for i = 1:N
    j = pol_H(i);
    M(i,j) = p;
    M(i,j+N) = 1-p;
    k = pol_L(i);
    M(i+N,k+N) = p;
    M(i+N,k) = 1-p;
end

[eigenvectors, ~] = eigs(sparse(M'), 1);
normalized_eigenvectors = eigenvectors / sum(eigenvectors);
capital_dist_H = normalized_eigenvectors(1:N);
capital_dist_L = normalized_eigenvectors(N+1:2*N);
net_capital_distribution = capital_dist_H + capital_dist_L;
Khat = K*net_capital_distribution;

end
