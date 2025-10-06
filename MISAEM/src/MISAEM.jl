module MISAEM

using LinearAlgebra
using Random
using Statistics
using Distributions
using ProgressMeter

export SAEMLogisticRegression, fit!, predict, predict_proba, likelihood_saem

"""
    SAEMLogisticRegression

Logistic regression model that handles missing data using SAEM algorithm.

# Fields
- `maxruns::Int`: Maximum number of SAEM iterations (default: 500)
- `tol_em::Float64`: Convergence tolerance for SAEM algorithm (default: 1e-7)
- `nmcmc::Int`: Number of MCMC iterations per SAEM step (default: 2)
- `tau::Float64`: Learning rate decay parameter (default: 1.0)
- `k1::Int`: Number of initial iterations with step size 1 (default: 50)
- `var_cal::Bool`: Whether to calculate variance estimates (default: true)
- `ll_obs_cal::Bool`: Whether to calculate observed data likelihood (default: true)
- `subsets::Union{Vector{Int}, Nothing}`: Subset of features to use in model (default: nothing)
- `random_state::Union{Int, Nothing}`: Random state for reproducibility (default: nothing)

# Model State (set after fitting)
- `coef::Vector{Float64}`: Model coefficients (including intercept)
- `intercept::Float64`: Model intercept
- `std_err::Vector{Float64}`: Standard errors of coefficients
- `mu::Vector{Float64}`: Estimated mean of covariates
- `sigma::Matrix{Float64}`: Estimated covariance matrix of covariates
- `ll_obs::Float64`: Observed data log-likelihood
- `converged::Bool`: Whether the algorithm converged
- `n_iterations::Int`: Number of iterations until convergence
"""
mutable struct SAEMLogisticRegression
    # Hyperparameters
    maxruns::Int
    tol_em::Float64
    nmcmc::Int
    tau::Float64
    k1::Int
    var_cal::Bool
    ll_obs_cal::Bool
    subsets::Union{Vector{Int}, Nothing}
    random_state::Union{Int, Nothing}
    
    # Model state (initialized after fitting)
    coef::Vector{Float64}
    intercept::Float64
    std_err::Vector{Float64}
    mu::Vector{Float64}
    sigma::Matrix{Float64}
    ll_obs::Float64
    converged::Bool
    n_iterations::Int
    
    function SAEMLogisticRegression(;
        maxruns::Int = 500,
        tol_em::Float64 = 1e-7,
        nmcmc::Int = 2,
        tau::Float64 = 1.0,
        k1::Int = 50,
        var_cal::Bool = true,
        ll_obs_cal::Bool = true,
        subsets::Union{Vector{Int}, Nothing} = nothing,
        random_state::Union{Int, Nothing} = nothing
    )
        
        # Validate parameters
        maxruns <= 0 && throw(ArgumentError("maxruns must be a positive integer."))
        tol_em <= 0 && throw(ArgumentError("tol_em must be a positive float."))
        nmcmc <= 0 && throw(ArgumentError("nmcmc must be a positive integer."))
        tau <= 0 && throw(ArgumentError("tau must be a positive float."))
        k1 < 0 && throw(ArgumentError("k1 must be a non-negative integer."))
        
        new(maxruns, tol_em, nmcmc, tau, k1, var_cal, ll_obs_cal, subsets, random_state,
            Float64[], 0.0, Float64[], Float64[], Matrix{Float64}(undef, 0, 0), 0.0, false, 0)
    end
end

"""
    check_X_y(X, y=nothing; predict=false)

Validate input data X and y.
"""
function check_X_y(X::AbstractMatrix, y::Union{AbstractVector, Nothing}=nothing; predict::Bool=false)
    if y === nothing && !predict
        throw(ArgumentError("y cannot be nothing when fitting."))
    end
    
    X_copy = Matrix{Float64}(X)
    y_copy = y === nothing ? nothing : Vector{Int}(y)
    
    if y_copy !== nothing
        if any(ismissing, y_copy) || any(isnan, y_copy)
            throw(ArgumentError("No missing data allowed in response variable y"))
        end
        
        unique_y = unique(y_copy)
        if length(unique_y) != 2 || !issetequal(unique_y, [0, 1])
            throw(ArgumentError("y must be binary with values 0 and 1."))
        end
    end
    
    if all(isnan, X_copy)
        throw(ArgumentError("X contains only NaN values."))
    end
    
    # Remove rows with all NaN values
    complete_rows = .!all(isnan.(X_copy), dims=2)[:]
    
    if any(.!complete_rows)
        sum_removed = sum(.!complete_rows)
        @warn "$(sum_removed) rows with all NaN values in X have been removed."
        
        if y_copy !== nothing
            y_copy = y_copy[complete_rows]
        end
        X_copy = X_copy[complete_rows, :]
    end
    
    if !predict
        if any(all(isnan.(X_copy), dims=1))
            throw(ArgumentError("X contains at least one column with only NaN values."))
        end
    end
    
    if y_copy !== nothing
        if size(X_copy, 1) != length(y_copy)
            throw(ArgumentError("Number of samples in X and y do not match."))
        end
    end
    
    return X_copy, y_copy
end

"""
    logistic_regression_fit(X, y)

Fit a simple logistic regression using iterative reweighted least squares.
"""
function logistic_regression_fit(X::AbstractMatrix, y::AbstractVector; max_iter::Int=1000, tol::Float64=1e-8)
    n, p = size(X)
    X_design = hcat(ones(n), X)
    beta = zeros(p + 1)
    
    for iter in 1:max_iter
        linear_pred = X_design * beta
        # Prevent overflow
        linear_pred = clamp.(linear_pred, -700, 700)
        prob = 1 ./ (1 .+ exp.(-linear_pred))
        
        # Avoid numerical issues
        prob = clamp.(prob, 1e-15, 1 - 1e-15)
        
        w = prob .* (1 .- prob)
        W = Diagonal(w)
        
        # Check for numerical issues
        if any(w .< 1e-15)
            @warn "Numerical instability detected in logistic regression"
            break
        end
        
        z = linear_pred .+ (y .- prob) ./ w
        
        try
            beta_new = (X_design' * W * X_design) \ (X_design' * W * z)
            
            if norm(beta_new - beta) < tol
                beta = beta_new
                break
            end
            
            beta = beta_new
        catch e
            @warn "Numerical error in logistic regression: $e"
            break
        end
    end
    
    return beta
end

"""
    stochastic_step!(X_sim, unique_patterns, pattern_indices, sigma_inv, mu, beta, y, nmcmc)

Perform the stochastic step of the SAEM algorithm.
"""
function stochastic_step!(X_sim::AbstractMatrix, unique_patterns::AbstractMatrix, 
                         pattern_indices::AbstractVector, sigma_inv::AbstractMatrix,
                         mu::AbstractVector, beta::AbstractVector, y::AbstractVector, nmcmc::Int)
    
    for (pattern_idx, pattern) in enumerate(eachrow(unique_patterns))
        if !any(pattern)
            continue
        end
        
        rows_with_pattern = findall(x -> x == pattern_idx, pattern_indices)
        n_pattern = length(rows_with_pattern)
        
        missing_idx = findall(pattern)
        obs_idx = findall(.!pattern)
        n_missing = length(missing_idx)
        
        if n_missing == 0
            continue
        end
        
        # Compute conditional MVN parameters
        Q_MM = sigma_inv[missing_idx, missing_idx]
        Q_MO = sigma_inv[missing_idx, obs_idx]
        
        sigma_cond_M = inv(Q_MM)
        
        if !isempty(obs_idx)
            X_O = X_sim[rows_with_pattern, obs_idx]
            delta_X_term = (X_O .- mu[obs_idx]')'
            adjustment_term = (sigma_cond_M * (Q_MO * delta_X_term))'
            mu_cond_M = mu[missing_idx]' .- adjustment_term
            
            lobs = beta[1] .+ X_O * beta[obs_idx .+ 1]
        else
            mu_cond_M = repeat(mu[missing_idx]', n_pattern, 1)
            lobs = fill(beta[1], n_pattern)
        end
        
        cobs = exp.(lobs)
        xina = X_sim[rows_with_pattern, missing_idx]
        betana = beta[missing_idx .+ 1]
        y_pattern = y[rows_with_pattern]
        
        # Cholesky decomposition for sampling
        chol_sigma_cond_M = cholesky(Hermitian(sigma_cond_M)).L
        
        for m in 1:nmcmc
            # Generate candidates
            rand_normal = randn(n_pattern, n_missing)
            xina_c = mu_cond_M .+ rand_normal * chol_sigma_cond_M'
            
            # Compute acceptance probabilities
            current_logit_contrib = sum(xina .* betana', dims=2)[:]
            candidate_logit_contrib = sum(xina_c .* betana', dims=2)[:]
            
            is_y1 = y_pattern .== 1
            
            ratio_y1 = (1 .+ exp.(-current_logit_contrib) ./ cobs) ./ 
                      (1 .+ exp.(-candidate_logit_contrib) ./ cobs)
            ratio_y0 = (1 .+ exp.(current_logit_contrib) .* cobs) ./ 
                      (1 .+ exp.(candidate_logit_contrib) .* cobs)
            
            alpha = ifelse.(is_y1, ratio_y1, ratio_y0)
            
            # Accept or reject
            accepted = rand(n_pattern) .< alpha
            xina[accepted, :] = xina_c[accepted, :]
        end
        
        X_sim[rows_with_pattern, missing_idx] = xina
    end
    
    return X_sim
end

"""
    louis_lr_saem(beta, mu, sigma, y, X_obs, pos_var, rindic, nmcmc)

Compute the Louis method for variance estimation in SAEM.
"""
function louis_lr_saem(beta::AbstractVector, mu::AbstractVector, sigma::AbstractMatrix,
                      y::AbstractVector, X_obs::AbstractMatrix, pos_var::AbstractVector,
                      rindic::AbstractMatrix, nmcmc::Int)
    
    n, p = size(X_obs)
    p_subset = length(pos_var)
    
    # Subset parameters
    beta_subset = beta[[1; pos_var .+ 1]]
    mu_subset = mu[pos_var]
    sigma_subset = sigma[pos_var, pos_var]
    X_subset = X_obs[:, pos_var]
    rindic_subset = rindic[:, pos_var]
    
    # Initialize matrices
    G = zeros(p_subset + 1, p_subset + 1)
    D = zeros(p_subset + 1, p_subset + 1)
    I_obs = zeros(p_subset + 1, p_subset + 1)
    Delta = zeros(p_subset + 1, 1)
    
    S_inv = inv(sigma_subset)
    
    # Initialize X_sim with mean imputation
    X_sim = copy(X_subset)
    for j in 1:p_subset
        nan_mask = isnan.(X_sim[:, j])
        if any(nan_mask)
            X_sim[nan_mask, j] .= mean(X_sim[.!nan_mask, j])
        end
    end
    
    for i in 1:n
        jna = findall(isnan.(X_subset[i, :]))
        njna = length(jna)
        
        if njna == 0
            x = vcat([1.0], X_sim[i, :])
            exp_b = exp(dot(beta_subset, x))
            d2l = -x * x' * (exp_b / (1 + exp_b)^2)
            I_obs -= d2l
        end

        if njna > 0
            xi = X_sim[i, :]
            
            # Extract submatrix and ensure symmetry
            S_inv_sub = S_inv[jna, jna]
            S_inv_sub = (S_inv_sub + S_inv_sub') / 2  # Force symmetry
            
            local Oi
            try
                Oi = inv(S_inv_sub)
            catch
                Oi = inv(S_inv_sub + 1e-6 * LinearAlgebra.I(njna))
            end
            
            mi = mu_subset[jna]
            lobs = beta_subset[1]
            
            if njna < p_subset
                jobs = setdiff(1:p_subset, jna)
                mi = mi - (Oi * S_inv[jna, jobs] * (xi[jobs] - mu_subset[jobs]))
                lobs += dot(xi[jobs], beta_subset[jobs .+ 1])
            end
            
            cobs = exp(lobs)
            xina = xi[jna]
            betana = beta_subset[jna .+ 1]
            
            for m in 1:nmcmc
                # Generate candidate - fix the random generation
                if njna == 1
                    # Scalar case
                    xina_c = mi .+ sqrt(Oi[1,1]) * randn()
                else
                    # Multivariate case - ensure Oi is positive definite
                    Oi_sym = (Oi + Oi') / 2
                    try
                        L = cholesky(Oi_sym).L
                        xina_c = mi + L * randn(njna)
                    catch
                        # Fallback: use eigendecomposition
                        eigen_decomp = eigen(Oi_sym)
                        # Keep only positive eigenvalues
                        pos_idx = eigen_decomp.values .> 1e-10
                        L = eigen_decomp.vectors[:, pos_idx] * 
                            Diagonal(sqrt.(eigen_decomp.values[pos_idx]))
                        xina_c = mi + L * randn(sum(pos_idx))
                    end
                end
                
                # Metropolis-Hastings step
                if y[i] == 1
                    alpha = (1 + exp(-dot(xina, betana)) / cobs) / 
                           (1 + exp(-dot(xina_c, betana)) / cobs)
                else
                    alpha = (1 + exp(dot(xina, betana)) * cobs) / 
                           (1 + exp(dot(xina_c, betana)) * cobs)
                end
                
                if rand() < alpha
                    xina = xina_c
                end
                
                X_sim[i, jna] = xina
                x = vcat([1.0], X_sim[i, :])
                exp_b = exp(dot(beta_subset, x))
                
                dl = x * (y[i] - exp_b / (1 + exp_b))
                d2l = -x * x' * (exp_b / (1 + exp_b)^2)
                
                D = D + (1/m) * (d2l - D)
                G = G + (1/m) * (dl * dl' - G)
                Delta = Delta + (1/m) * (dl - Delta)
            end
            
            I_obs -= D + G - Delta * Delta'
        end
    end
    
    V_obs = inv(I_obs)
    return V_obs
end

"""
    likelihood_saem(beta, mu, sigma, y, X_obs, rindic, nmcmc)

Compute the observed data log-likelihood using SAEM.
"""
function likelihood_saem(beta::AbstractVector, mu::AbstractVector, sigma::AbstractMatrix,
                        y::AbstractVector, X_obs::AbstractMatrix, rindic::AbstractMatrix, nmcmc::Int)
    
    n, p = size(X_obs)
    lh = 0.0
    
    for i in 1:n
        yi = y[i]
        xi = X_obs[i, :]
        
        if !any(rindic[i, :])
            # No missing values
            x_design = vcat([1.0], xi)
            lh += yi * dot(beta, x_design) - log(1 + exp(dot(beta, x_design)))
        else
            # Missing values - use MCMC integration
            miss_col = findall(rindic[i, :])
            obs_col = findall(.!rindic[i, :])
            
            if isempty(obs_col)
                # All missing
                mu_cond = mu[miss_col]
                sigma_cond = sigma[miss_col, miss_col]
            else
                # Partial missing
                x2 = xi[obs_col]
                mu1 = mu[miss_col]
                mu2 = mu[obs_col]
                
                sigma11 = sigma[miss_col, miss_col]
                sigma12 = sigma[miss_col, obs_col]
                sigma22 = sigma[obs_col, obs_col]
                
                mu_cond = mu1 + sigma12 * (sigma22 \ (x2 - mu2))
                sigma_cond = sigma11 - sigma12 * (sigma22 \ sigma12')
            end
            
            # MCMC integration
            lh_mis = 0.0
            for m in 1:nmcmc
                # Generate missing values
                x1_sample = mu_cond + randn(length(miss_col))' * cholesky(sigma_cond).L'
                
                # Complete the observation
                xi_complete = copy(xi)
                xi_complete[miss_col] = x1_sample
                
                x_design = vcat([1.0], xi_complete)
                linear_pred = dot(beta, x_design)
                lh_mis += exp(yi * linear_pred - log(1 + exp(linear_pred)))
            end
            
            lh += log(lh_mis / nmcmc)
        end
    end
    
    return lh
end

"""
    fit!(model::SAEMLogisticRegression, X::AbstractMatrix, y::AbstractVector; 
         save_trace=false, progress_bar=true)

Fit the SAEM logistic regression model.
"""
function fit!(model::SAEMLogisticRegression, X::AbstractMatrix, y::AbstractVector; 
              save_trace::Bool=false, progress_bar::Bool=true)
    
    # Set random seed if specified
    if model.random_state !== nothing
        Random.seed!(model.random_state)
    end
    
    # Validate inputs
    X_clean, y_clean = check_X_y(X, y)
    n, p = size(X_clean)
    
    # Set subsets
    if model.subsets === nothing
        subsets = 1:p
    else
        subsets = model.subsets
        if length(unique(subsets)) != length(subsets)
            throw(ArgumentError("Subsets must be unique."))
        end
    end
    
    # Check for missing data
    rindic = isnan.(X_clean)
    missing_cols = any(rindic, dims=1)[:]
    num_missing_cols = sum(missing_cols)
    
    if num_missing_cols > 0
        # Initialize with mean imputation
        X_sim = copy(X_clean)
        for j in 1:p
            nan_mask = isnan.(X_sim[:, j])
            if any(nan_mask)
                X_sim[nan_mask, j] .= mean(X_sim[.!nan_mask, j])
            end
        end
        
        # Initialize parameters
        mu = vec(mean(X_sim, dims=1))
        sigma = cov(X_sim) * (n-1) / n
        sigma_inv = inv(sigma)
        
        # Initial logistic regression
        beta_lr = logistic_regression_fit(X_sim[:, subsets], y_clean)
        beta = zeros(p + 1)
        beta[[1; subsets .+ 1]] = beta_lr
        
        # Find unique missing patterns
        unique_patterns = unique(rindic, dims=1)
        pattern_indices = [findfirst(x -> all(x .== row), eachrow(unique_patterns)) 
                          for row in eachrow(rindic)]
        
        # SAEM iterations
        prog = progress_bar ? Progress(model.maxruns, desc="SAEM Iterations: ") : nothing
        
        for k in 1:model.maxruns
            beta_old = copy(beta)
            
            # Stochastic step
            X_sim = stochastic_step!(X_sim, unique_patterns, pattern_indices, 
                                   sigma_inv, mu, beta, y_clean, model.nmcmc)
            
            # Maximization step
            beta_lr_new = logistic_regression_fit(X_sim[:, subsets], y_clean)
            beta_new = zeros(p + 1)
            beta_new[[1; subsets .+ 1]] = beta_lr_new
            
            # Update with step size
            gamma = k <= model.k1 ? 1.0 : 1.0 / ((k - model.k1)^model.tau)
            beta = (1 - gamma) * beta + gamma * beta_new
            mu = (1 - gamma) * mu + gamma * vec(mean(X_sim, dims=1))
            sigma_new = cov(X_sim, corrected=false)
            sigma = (1 - gamma) * sigma + gamma * sigma_new
            sigma_inv = inv(sigma)
            
            # Check convergence
            if sum((beta - beta_old).^2) < model.tol_em
                model.converged = true
                model.n_iterations = k
                progress_bar && println("\n...converged after $k iterations.")
                break
            end
            
            progress_bar && next!(prog)
        end
        
        if !model.converged
            model.n_iterations = model.maxruns
        end
        
        # Compute variance if requested
        if model.var_cal
            var_obs = louis_lr_saem(beta, mu, sigma, y_clean, X_clean, 
                                  collect(subsets), rindic, 100)
            model.std_err = sqrt.(diag(var_obs))
        end
        
        # Compute likelihood if requested
        if model.ll_obs_cal
            model.ll_obs = likelihood_saem(beta, mu, sigma, y_clean, X_clean, rindic, 100)
        end
        
    else
        # No missing data - standard logistic regression
        beta_lr = logistic_regression_fit(X_clean, y_clean)
        beta = beta_lr
        mu = vec(mean(X_clean, dims=1))
        sigma = cov(X_clean) * (n-1) / n
        
        model.converged = true
        model.n_iterations = 0
        
        # Compute variance
        if model.var_cal
            X_design = hcat(ones(n), X_clean)
            linear_pred = X_design * beta
            prob = 1 ./ (1 .+ exp.(-linear_pred))
            W = Diagonal(prob .* (1 .- prob))
            var_obs = inv(X_design' * W * X_design)
            model.std_err = sqrt.(diag(var_obs))
        end
        
        # Compute likelihood
        if model.ll_obs_cal
            model.ll_obs = likelihood_saem(beta, mu, sigma, y_clean, X_clean, 
                                         zeros(Bool, size(X_clean)), 100)
        end
    end
    
    # Store final parameters
    final_params = beta[[1; subsets .+ 1]]
    model.intercept = final_params[1]
    model.coef = final_params
    model.mu = mu
    model.sigma = sigma
    
    return model
end

"""
    predict_proba(model::SAEMLogisticRegression, X_test::AbstractMatrix; 
                  method=:map, nmcmc=500, random_state=nothing)

Predict class probabilities for samples in X_test.
"""
function predict_proba(model::SAEMLogisticRegression, X_test::AbstractMatrix;
                      method::Symbol=:map, nmcmc::Int=500, random_state::Union{Int,Nothing}=nothing)
    
    if random_state !== nothing
        Random.seed!(random_state)
    end
    
    X_test_clean, _ = check_X_y(X_test; predict=true)
    n, p = size(X_test_clean)
    
    subsets = model.subsets === nothing ? (1:p) : model.subsets
    beta_saem = model.coef
    mu_saem = model.mu
    sigma_saem = model.sigma
    
    pr_saem = zeros(n)
    rindic = isnan.(X_test_clean)
    
    # Find unique patterns
    unique_patterns = unique(rindic, dims=1)
    pattern_indices = [findfirst(x -> all(x .== row), eachrow(unique_patterns)) 
                      for row in eachrow(rindic)]
    
    for (pattern_idx, pattern) in enumerate(eachrow(unique_patterns))
        rows_with_pattern = findall(x -> x == pattern_idx, pattern_indices)
        isempty(rows_with_pattern) && continue
        
        xi_pattern = X_test_clean[rows_with_pattern, :]
        
        if !any(pattern)
            # No missing values
            X_subset = xi_pattern[:, subsets]
            X_design = hcat(ones(length(rows_with_pattern)), X_subset)
            linear_pred = X_design * beta_saem
            pr_saem[rows_with_pattern] = 1 ./ (1 .+ exp.(-linear_pred))
            continue
        end
        
        miss_col = findall(pattern)
        obs_col = findall(.!pattern)
        
        if method == :impute
            # Simple conditional mean imputation
            mu1 = mu_saem[miss_col]
            
            if !isempty(obs_col)
                mu2 = mu_saem[obs_col]
                sigma12 = sigma_saem[miss_col, obs_col]
                sigma22 = sigma_saem[obs_col, obs_col]
                
                x2 = xi_pattern[:, obs_col]
                mu_cond = mu1' .+ (x2 .- mu2') * (sigma22 \ sigma12')
                X_test_clean[rows_with_pattern, miss_col] = mu_cond
            else
                X_test_clean[rows_with_pattern, miss_col] .= mu1'
            end
            
        elseif method == :map
            # MCMC marginalization
            n_pattern = length(rows_with_pattern)
            mu1 = mu_saem[miss_col]
            
            if !isempty(obs_col)
                mu2 = mu_saem[obs_col]
                sigma11 = sigma_saem[miss_col, miss_col]
                sigma12 = sigma_saem[miss_col, obs_col]
                sigma22 = sigma_saem[obs_col, obs_col]
                
                sigma_cond = sigma11 - sigma12 * (sigma22 \ sigma12')
                chol_sigma_cond = cholesky(Hermitian(sigma_cond)).L
                
                x2 = xi_pattern[:, obs_col]
                solve_term = (x2 .- mu2') * (sigma22 \ sigma12')
                mu_cond = mu1' .+ solve_term
            else
                sigma_cond = sigma_saem[miss_col, miss_col]
                chol_sigma_cond = cholesky(Hermitian(sigma_cond)).L
                mu_cond = repeat(mu1', n_pattern, 1)
            end
            
            # MCMC integration
            probs = zeros(n_pattern)
            for i in 1:n_pattern
                prob_sum = 0.0
                for m in 1:nmcmc
                    # Generate missing values
                    rand_sample = randn(length(miss_col))
                    x1_sample = mu_cond[i, :] + chol_sigma_cond * rand_sample
                    
                    # Complete observation
                    xi_complete = copy(xi_pattern[i, :])
                    xi_complete[miss_col] = x1_sample
                    
                    # Compute probability
                    X_subset = xi_complete[subsets]
                    x_design = vcat([1.0], X_subset)
                    linear_pred = dot(beta_saem, x_design)
                    prob_sum += 1 / (1 + exp(-linear_pred))
                end
                probs[i] = prob_sum / nmcmc
            end
            
            pr_saem[rows_with_pattern] = probs
            
        else
            throw(ArgumentError("Method must be either :impute or :map"))
        end
    end
    
    if method == :impute
        # Recompute probabilities after imputation
        X_subset = X_test_clean[:, subsets]
        X_design = hcat(ones(n), X_subset)
        linear_pred = X_design * beta_saem
        pr_saem = 1 ./ (1 .+ exp.(-linear_pred))
    end
    
    return hcat(1 .- pr_saem, pr_saem)
end

"""
    predict(model::SAEMLogisticRegression, X_test::AbstractMatrix; kwargs...)

Predict class labels for samples in X_test.
"""
function predict(model::SAEMLogisticRegression, X_test::AbstractMatrix; kwargs...)
    probs = predict_proba(model, X_test; kwargs...)
    return Int.(probs[:, 2] .>= 0.5)
end

end # module