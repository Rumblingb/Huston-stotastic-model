function mainFinal()
    % Parameter Definition
    S0 = 100;       
    v0 = 0.09;      
    r = 0.05;       
    theta = 0.348;  
    sigma = 0.39;   
    kappa = 1.15;   
    rho = -0.64;    
    K = [90, 100, 110];  
    T = [0.5, 1.0, 2.0]; 
    n_values = [10, 20, 30]; 
    upper_bound = 1e5;       
    
    % 'results' will store rows of [T, K, n, Price, CPUtime]
    results = [];
    
    % Option prices can be stored in 'prices' if needed
    prices = zeros(length(T), length(K), length(n_values));
    
    fprintf('==== Geometric Asian Call Option Prices ====\n');
    
    for t_idx = 1:length(T)
        for k_idx = 1:length(K)
            for n_idx = 1:length(n_values)
                % Record CPU time
                startTime = tic;
                price = compute_asian_price(...
                    S0, v0, r, theta, sigma, kappa, rho, ...
                    K(k_idx), T(t_idx), n_values(n_idx), upper_bound);
                cpuTime = toc(startTime);
                
                prices(t_idx, k_idx, n_idx) = price;
                % Store [T, K, n, Price, CPU time] in 'results'
                results = [results; T(t_idx), K(k_idx), n_values(n_idx), price, cpuTime];
            end
        end
    end
    
    % Display results in the Command Window
    disp('Geometric Asian Call Option Prices (T, K, n, Price, CPUtime):');
    disp(results);
    
    % Export results to Excel
    varNames = {'T','K','n','OptionValue','CPUtime'};
    TBL = array2table(results, 'VariableNames', varNames);
    writetable(TBL, 'GeometricAsianCallTable.xlsx');
    fprintf('\nResults have been saved to GeometricAsianCallTable.xlsx\n');
end

function price = compute_asian_price(S0, v0, r, theta, sigma, kappa, rho, K, T, N_terms, upper_bound)
    % Compute the price of a fixed-strike geometric Asian call option
    % under the Heston model using Fourier inversion and the joint
    % characteristic function.
    
    % Define parameters a1-a5
    a1 = 2 * v0 / sigma^2;
    a2 = 2 * kappa * theta / sigma^2;
    a3 = log(S0) + ((r * sigma - kappa * theta * rho) * T) / (2 * sigma) - (rho * v0 / sigma);
    a4 = log(S0) - (rho * v0 / sigma) + (r - (rho * kappa * theta / sigma)) * T;
    a5 = (kappa * v0 + kappa^2 * theta * T) / sigma^2;
    
    % Define psi_t(s, w)
    psi = @(s, w) exp( ...
        -a1 * (compute_H_Htilde(s, w, T, N_terms, kappa, sigma, rho, true) ...
               / compute_H_Htilde(s, w, T, N_terms, kappa, sigma, rho, false)) ...
        - a2 * log(compute_H_Htilde(s, w, T, N_terms, kappa, sigma, rho, false)) ...
        + a3 * s + a4 * w + a5 );
    
    % Define the integrand for the Fourier inversion
    integrand = @(xi) real( ...
        (psi(1 + 1i*xi, 0) - K * psi(1i*xi, 0)) ...
        .* exp(-1i * xi * log(K)) ./ (1i * xi) );
    
    % Numerical integration (using Matlab's integral function)
    integral_value = (1/pi) * integral(integrand, 0, upper_bound, ...
        'ArrayValued', true, 'RelTol', 1e-4);
    
    % Final price of the Asian call option (discounted)
    price = exp(-r * T) * ((psi(1, 0) - K)/2 + integral_value);
end

function [H, H_tilde] = compute_H_Htilde(s, w, T, N_terms, kappa, sigma, rho, return_H_tilde)
    % This function computes H and H_tilde based on the recursive definition
    % of h_n(s, w). If return_H_tilde == true, we return H_tilde instead of H.
    
    % Ensure N_terms is an integer
    N_terms = round(N_terms);
    if N_terms < 1
        error('N_terms must be an integer >= 1');
    end
    
    % Allocate h array: h(-2), h(-1), h(0), ..., h(N_terms)
    h = zeros(1, N_terms + 3);
    h(1) = 0;   % h(-2)
    h(2) = 0;   % h(-1)
    h(3) = 1;   % h(0)
    if N_terms >= 1
        h(4) = (T * (kappa - w * rho * sigma)) / 2; % h(1)
    end
    
    % Recursively compute h(n) for n = 2,3,...,N_terms
    for n = 2:N_terms
        idx = n + 3;
        term1 = -s^2 * sigma^2 * (1 - rho^2) * T^2;
        term2 = T * (s * sigma * T * (sigma - 2*rho*kappa) ...
            - 2*s*w*sigma^2*T*(1 - rho^2));
        term3 = T * (kappa^2 * T - 2*s*rho*sigma ...
            - w*(2*rho*kappa - sigma)*sigma*T ...
            - w^2*(1 - rho^2)*sigma^2*T);
        
        % Use previous terms: h(n-4), h(n-3), h(n-2)
        if n-4 >= 0
            h_n4 = h(n - 4 + 3);
        else
            h_n4 = 0;
        end
        if n-3 >= 0
            h_n3 = h(n - 3 + 3);
        else
            h_n3 = 0;
        end
        if n-2 >= 0
            h_n2 = h(n - 2 + 3);
        else
            h_n2 = 0;
        end
        
        h(idx) = (term1 * h_n4 + term2 * h_n3 + term3 * h_n2) / (4 * n * (n-1));
    end
    
    % Sum h(0) through h(N_terms)
    H = sum(h(3:3 + N_terms));
    H_tilde = sum((1:N_terms) .* h(4:3 + N_terms)) / T;
    
    % Return H_tilde or H based on return_H_tilde
    if return_H_tilde
        H = H_tilde;
    end
end
