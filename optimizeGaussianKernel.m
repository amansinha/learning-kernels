function [W_opt, b_opt, alpha, alpha_distrib] = optimizeGaussianKernel(Xtrain, ytrain, Nw, rho, tol)
% OPTIMIZEGAUSSIANKERNEL optimizes random features generated for the
% Gaussian kernel using the chi-square divergence measure.
% See http://amansinha.org/docs/SinhaDu16.pdf for more info on the theory.
% Inputs:
% Xtrain is the d x N training data matrix, where N is the number of 
%    datapoints and d is the dimension.
% ytrain is the N x 1 training label vector. The binary classes should be 1
%     and -1.
% Nw is the number of random features to use.
% rho governs the maximum allowable divergence form the original kernel
%     distribution
% tol is the tolerance for the solver.
%
% Outputs: 
% W_opt is the optimized matrix of random features
% b_opt is the optimized vector of offsets
% alpha is the probability distribution for the random features
%     with close-to-zero-probability features removed
% alpha_distrib is cumulative distribution function over all random
%     features
    
    [d, ~] = size(Xtrain);
    
    % generate standard Gaussian random features
    W = randn(d, Nw); 
    b = rand(1,Nw)*2*pi;
    
    % set up/solve the problem using the chi-square divergence
    Phi = cos(bsxfun(@plus,Xtrain'*W, b));
    Ks = Phi'*ytrain;
    Ks = Ks.^2;
    alpha_temp = linear_chi_square(-Ks, 1/Nw*ones(Nw,1), rho/Nw, tol);
    
    % grab the non-zero-probability features
    idx = alpha_temp > eps;
    alpha = alpha_temp(idx);
    W_opt = W(:,idx);
    b_opt = b(idx);
    alpha_distrib = cumsum(alpha/sum(alpha));
end