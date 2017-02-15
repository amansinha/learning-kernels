%% Learning kernels with Random Features, example script 2

% Generate some data
% Create normally distributed points and let the true classifier be a
% certain hyperplane
clear all;
close all;
clc;
rng(7117);

n = 10000;
d = 1000;
Xtrain = randn(d, n);
Xtest = randn(d, n/10);
% the first few dimensions are much more important than the rest
true_vec = [ones(d/10,1); 0.05*randn(d-d/10,1)];

ytrain = Xtrain'*true_vec > 0;
ytrain = ytrain*2-1;
ytest = Xtest'*true_vec > 0;
ytest = ytest*2-1;

%% Optimize kernel
% This example uses the linear kernel and chi-square divergence
rho = d*0.01;
tol = 1e-11;
[idxopt, alpha, alpha_distrib] = optimizeLinearKernel(Xtrain, ytrain, rho, tol);
% Take a look at what the distirbution looks like
figure
plot(sort(alpha))
xlabel('Feature')
ylabel('Probability')

%% Create random features using optimized kernel
% pick a number of random features to use for the model
D = length(alpha);
% generate parameters for the optimized kernel
[D_opt, idx_opt] = createOptimizedLinearKernelParams(D, idxopt, alpha_distrib);
% create optimized features using the training data and test data
Z_opt_train = Xtrain(idx_opt, :);
Z_opt_test = Xtest(idx_opt, :);

% Generate regular random linear features for comparison
idx = randperm(d, D);
Z_train = Xtrain(idx, :);
Z_test = Xtest(idx, :);

%% Train models
% For simplicity, train linear regression models (even though this is a
% classification problem!)
meany = mean(ytrain);
lambda = .01;
w_opt = (Z_opt_train * Z_opt_train' + lambda * eye(D_opt)) \ (Z_opt_train * (ytrain-meany));
w = (Z_train * Z_train' + lambda * eye(D)) \ (Z_train * (ytrain-meany));
% Note that we don't bother scaling the features by sqrt(alpha) since we
% can absorb that factor into w_opt for this ridge regression model

% If you have the right packages, you can drop in smarter models easily:
% mdl = fitglm(Z_train', (ytrain+1)/2, 'Distribution', 'binomial');
% or
% mdl = fitcsvm(Z_train', ytrain, 'KernelFunction', 'linear', 'ClassNames', [-1, 1]);
% and then change the error computation code accordingly for the logistic
% regression or SVM models respectively.

%% errors
%calculate errors on training set
disp(['Fraction of positives (train): ' num2str(sum(ytrain==1)/length(ytrain))])
[err,fp, fn] = computeError(Z_train, w, meany, ytrain);
disp(['Regular train error: ' num2str(err)]);
disp(['false positives: ' num2str(fp)]);
disp(['false negatives: ' num2str(fn)]);
disp(' ')
[err,fp, fn] = computeError(Z_opt_train, w_opt, meany, ytrain);
disp(['Optimized train error: ' num2str(err)])
disp(['false positives: ' num2str(fp)])
disp(['false negatives: ' num2str(fn)])
disp(' ')
disp(' ')

disp(['Fraction of positives (test): ' num2str(sum(ytest==1)/length(ytest))])
[err,fp, fn] = computeError(Z_test, w, meany, ytest);
disp(['Regular test error: ' num2str(err)]);
disp(['false positives: ' num2str(fp)]);
disp(['false negatives: ' num2str(fn)]);
disp(' ')
[err,fp, fn] = computeError(Z_opt_test, w_opt, meany, ytest);
disp(['Optimized test error: ' num2str(err)])
disp(['false positives: ' num2str(fp)])
disp(['false negatives: ' num2str(fn)])

% This should generate something like the following
% Fraction of positives (train): 0.5
% Regular train error: 0.3738
% false positives: 0.49652
% false negatives: 0.50348
%  
% Optimized train error: 0.047
% false positives: 0.51915
% false negatives: 0.48085
%  
%  
% Fraction of positives (test): 0.501
% Regular test error: 0.373
% false positives: 0.52279
% false negatives: 0.47721
%  
% Optimized test error: 0.051
% false positives: 0.64706
% false negatives: 0.35294