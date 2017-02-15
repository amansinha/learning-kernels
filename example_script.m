%% Learning kernels with Random Features, example script

% Generate some data
% Create normally distributed points and let the true classifier between
% classes be a specified radius
clear all;
close all;
clc;
rng(7117);

n = 10000;
d = 10;
Xtrain = randn(d, n);
Xtest = randn(d, n/10);

ytrain = sqrt(sum(Xtrain.*Xtrain,1))>sqrt(d);
ytrain = ytrain'*2-1;
ytest = sqrt(sum(Xtest.*Xtest,1))>sqrt(d);
ytest = ytest'*2-1;

%% Optimize kernel
% This example uses the Gaussian kernel and chi-square divergence
Nw = 2e4;
rho = Nw*0.005;
tol = 1e-11;
[Wopt, bopt, alpha, alpha_distrib] = optimizeGaussianKernel(Xtrain, ytrain, Nw, rho, tol);
% Take a look at what the distirbution looks like
figure
plot(sort(alpha))
xlabel('Feature')
ylabel('Probability')

%% Create random features using optimized kernel
% pick a number of random features to use for the model
D = length(alpha);
% generate parameters for the optimized kernel
[D_opt, W_opt, b_opt] = createOptimizedGaussianKernelParams(D, Wopt, bopt, alpha_distrib);
% create optimized features using the training data and test data
Z_opt_train = createRandomFourierFeatures(D_opt, W_opt, b_opt, Xtrain);
Z_opt_test = createRandomFourierFeatures(D_opt, W_opt, b_opt, Xtest);

% Generate regular Gaussian features for comparison
W = randn(d,D);
b = rand(1,D)*2*pi;
Z_train = createRandomFourierFeatures(D, W, b, Xtrain);
Z_test = createRandomFourierFeatures(D, W, b, Xtest);

%% Train models
% For simplicity, train linear regression models (even though this is a
% classification problem!)
meany = mean(ytrain);
lambda = .05;
w_opt = (Z_opt_train * Z_opt_train' + lambda * eye(D_opt)) \ (Z_opt_train * (ytrain-meany));
w = (Z_train * Z_train' + lambda * eye(D)) \ (Z_train * (ytrain-meany));
% Note that we don't bother scaling the features by sqrt(alpha) since we
% can absorb that factor into w_opt for this ridge regression model

% If you have the ability to use smarter models, then you can try:
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
% Fraction of positives (train): 0.4381
% Regular train error: 0.3127
% false positives: 0.1308
% false negatives: 0.8692
%  
% Optimized train error: 0.1199
% false positives: 0.44621
% false negatives: 0.55379
%  
%  
% Fraction of positives (test): 0.457
% Regular test error: 0.355
% false positives: 0.12113
% false negatives: 0.87887
%  
% Optimized test error: 0.143
% false positives: 0.37762
% false negatives: 0.62238