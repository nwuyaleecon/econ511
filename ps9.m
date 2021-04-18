clc
clear
close all

%% all_code
% each problem part is its own function, at the bottom of the file
% WARNING:
% this will spit out a lot of figures--
% for sake of sanity, perhaps comment out parts 9-11 so that the code spits
% out fewer figuress

[all_data] = part1('Data.csv');
[names, N_vars, T, restricted_data] = part2(all_data);
data = part3(restricted_data, N_vars, T);
[X, B, U, A0] = part4(data, 2, N_vars, T);
[irf, C] = part5(B, A0, 2, N_vars, 16); 
[all_irfs] = part6(X, B, C, U, 2, N_vars, T, 1000, 16);
[irf_upper_quantiles, irf_lower_quantiles] = part7(irf, all_irfs, N_vars, 16, 0.1, 0.9, names);
part9('Data.csv');
part10('Data.csv');
part11('Data.csv');


%% part 1
function [all_data] = part1(data_file)
    all_data = readtable(data_file);
end

%% part 2
function [names, N_vars, T, restricted_data] = part2(all_data)
    disp(all_data.Properties.VariableNames(8)); % choosing M1
    restricted_data = [all_data{:,2:7}, all_data{:,8}]; % only use the chosen M_AGG, drop date
    N_vars = size(restricted_data, 2);
    T = size(restricted_data, 1);
    names = all_data.Properties.VariableNames(2:8);
end

%% part 3
function [data] = part3(restricted_data, N_vars, T)
    % detrend data
    data = detrend_data(restricted_data);
     % hpfilter from econometrics package

    % now we run equation by equation OLS:
    max_lags = 5;
    for p = 2:max_lags
        X = zeros(T-p,p*N_vars);
        for i = 1:p
            X(:,(i-1)*N_vars+1:i*N_vars) = data(p+1-i:T-i,:);
        end
    
        all_R2_vals = zeros(1,N_vars);
        for i = 1:N_vars
            [~, ~, ~, ~, stats] = regress(data(p+1:T,i), [ones(T-p,1), X]);
            all_R2_vals(1,i) = stats(1);
        end
        disp([num2str(p), ' lags R^2 values: ']);
        disp(all_R2_vals);
    end

% note that as the number of lags increases, the R2 values also increase
end

function [data] = detrend_data(restricted_data)
    data = restricted_data - hpfilter(restricted_data, 1600);
end

%% part 4
function [X, B, U, A0] = part4(data, p, K, T)

    % fix the lags
    X = zeros(T-p,p*K);
    for i = 1:p
        X(:,(i-1)*K+1:i*K) = data(p+1-i:T-i,:);
    end

    B = zeros(K,size(X, 2)+1); % +1 for the constant 
    U = zeros(K, T-p);
    for i = 1:K
        [coeffs, ~, residuals, ~, ~] = regress(data(p+1:T,i), [ones(T-p,1), X]);
        B(i,:) = coeffs;
        U(i,:) = residuals;
    end

    V = U*U'/(T-p);
    A0 = chol(V, 'lower');
end

%% part 5
function [irf, C] = part5(B, A0, p, K, IRF_T, shock)
    if nargin < 6
        shock = [0,0,0,0.01,0,0,0];
    end
    C = [B(:,2:p*K+1); eye((p-1)*K), zeros((p-1)*K, K)];
    impulse = [shock*A0',zeros(1,(p-1)*K)]';

    response = zeros(p*K, IRF_T);
    for t = 1:IRF_T
        response(:,t) = (C^(t-1))*impulse;
    end

    irf = response(1:K,:);
end

%% part 6

function [all_irfs] = part6(X, B, C, U, p, K, T, N_draws, IRF_T, shock)
    if nargin < 10
        shock = [0,0,0,0.01,0,0,0];
    end

    y_init = X(1,:)';

    sample_indices = 1 + floor( (T-p)*rand( N_draws , T-p ) ) ;
    
    sampled_y = zeros(T-p, K);
    all_irfs = zeros(N_draws, K, IRF_T);
    for j = 1:N_draws
        prev_y = y_init;
        for t = 1:(T-p)
            constant = [B(:,1);zeros((p-1)*K,1)];
            estimated_value = C*prev_y;
            sampled_error = [U(:,sample_indices(j,t)); zeros((p-1)*K,1)];
            next_y = constant + estimated_value + sampled_error;
            sampled_y(t, :) = next_y(1:K);
            prev_y = next_y;
        end
        
        [~, B_j, ~, A0_j] = part4(sampled_y, p, K, T-p);
        [irf_j, ~] = part5(B_j, A0_j, p, K, IRF_T, shock);
        all_irfs(j,:,:) = irf_j;
    end
    
end

%% part 7
function [irf_upper_quantiles, irf_lower_quantiles] = part7(irf, all_irfs, K, IRF_T, lower_cutoff, upper_cutoff, names)

    irf_upper_quantiles = zeros(K, IRF_T);
    irf_lower_quantiles = zeros(K, IRF_T);
    for t = 1:IRF_T 
        for k = 1:K
            irf_upper_quantiles(k, t) = quantile(all_irfs(:,k,t), upper_cutoff);
            irf_lower_quantiles(k, t) = quantile(all_irfs(:,k,t), lower_cutoff);
        end
    end
    
    for k = 1:K
        figure()
        plot(0:IRF_T-1, irf_upper_quantiles(k,:), '-')
        hold on
        plot(0:IRF_T-1, irf_lower_quantiles(k,:), '-')
        hold on
        plot(0:IRF_T-1, irf(k,:), '-')
        grid minor
        hold off
        legend('upper quantile','lower quantile','irf')
        title(names(k))
    end
    
end

%% part 9

function [] = part9(data_file) 
    
    [all_data] = part1(data_file);
    [names, N_vars, T, restricted_data] = part2(all_data);
    
    % first half
    data = detrend_data(restricted_data(1:T/2,:));
    [X, B, U, A0] = part4(data, 2, N_vars, T/2);
    [irf, C] = part5(B, A0, 2, N_vars, 16); 
    [all_irfs] = part6(X, B, C, U, 2, N_vars, T/2, 1000, 16);
    [~, ~] = part7(irf, all_irfs, N_vars, 16, 0.1, 0.9, names);
    
    % second half
    data = detrend_data(restricted_data(1 + T/2:T,:));
    [X, B, U, A0] = part4(data, 2, N_vars, T/2);
    [irf, C] = part5(B, A0, 2, N_vars, 16); 
    [all_irfs] = part6(X, B, C, U, 2, N_vars, T/2, 1000, 16);
    [~, ~] = part7(irf, all_irfs, N_vars, 16, 0.1, 0.9, names);
    
    % without first and last 10 years
    data = detrend_data(restricted_data(11:T-10,:));
    [X, B, U, A0] = part4(data, 2, N_vars, T-20);
    [irf, C] = part5(B, A0, 2, N_vars, 16); 
    [all_irfs] = part6(X, B, C, U, 2, N_vars, T-20, 1000, 16);
    [~, ~] = part7(irf, all_irfs, N_vars, 16, 0.1, 0.9, names);
end

function [] = part10(data_file) 
    
    [all_data] = part1(data_file);
    [names, N_vars, T, restricted_data] = part2(all_data);
    % large shock
    data = detrend_data(restricted_data);
    [X, B, U, A0] = part4(data, 2, N_vars, T);
    [irf, C] = part5(B, A0, 2, N_vars, 16, [0,0,0,0.05,0,0,0]); 
    [all_irfs] = part6(X, B, C, U, 2, N_vars, T, 1000, 16, [0,0,0,0.05,0,0,0]);
    [~, ~] = part7(irf, all_irfs, N_vars, 16, 0.1, 0.9, names);
    
end

function [] = part11(data_file)

    [all_data] = part1(data_file);
    [names, N_vars, T, restricted_data] = part2(all_data);
    
    swapped_data = restricted_data;
    swapped_data(:,2) = restricted_data(:,4);
    swapped_data(:,4) = restricted_data(:,2);
    swapped_names = names;
    swapped_names(2) = names(4);
    swapped_names(4) = names(2);
    
    % ff shock
    data = detrend_data(restricted_data);
    [X, B, U, A0] = part4(data, 2, N_vars, T);
    [irf, C] = part5(B, A0, 2, N_vars, 16);
    [all_irfs] = part6(X, B, C, U, 2, N_vars, T, 1000, 16);
    [~, ~] = part7(irf, all_irfs, N_vars, 16, 0.1, 0.9, names);
    % price shock
    [irf, C] = part5(B, A0, 2, N_vars, 16, [0,0.01,0,0,0,0,0]);
    [all_irfs] = part6(X, B, C, U, 2, N_vars, T, 1000, 16, [0,0.01,0,0,0,0,0]);
    [~, ~] = part7(irf, all_irfs, N_vars, 16, 0.1, 0.9, names);
    
    % price shock
    data = detrend_data(swapped_data);
    [X, B, U, A0] = part4(data, 2, N_vars, T);
    [irf, C] = part5(B, A0, 2, N_vars, 16);
    [all_irfs] = part6(X, B, C, U, 2, N_vars, T, 1000, 16);
    [~, ~] = part7(irf, all_irfs, N_vars, 16, 0.1, 0.9, swapped_names);
    % ff shock
    [irf, C] = part5(B, A0, 2, N_vars, 16, [0,0.01,0,0,0,0,0]);
    [all_irfs] = part6(X, B, C, U, 2, N_vars, T, 1000, 16, [0,0.01,0,0,0,0,0]);
    [~, ~] = part7(irf, all_irfs, N_vars, 16, 0.1, 0.9, swapped_names);
    
end