%% Gaussian Process Regression with OOD Region
% This code illustrates:
%   1. Training data in a data-rich region
%   2. GP mean prediction
%   3. GP epistemic uncertainty (+/- 2 sigma)
%   4. Out-of-distribution (OOD) region
%
% Requires Statistics and Machine Learning Toolbox

clear;
clc;
close all;

%% ============================================================
% 1. Generate training data
% =============================================================

rng(10);   % Reproducibility

% True nonlinear function
f = @(x) 0.55 + 0.30*sin(3*x);

% Training data only in the data-rich region
Xtrain = linspace(-3, 1.3, 45)';

% Measurement noise
noise_std = 0.05;

% Observed training outputs
Ytrain = f(Xtrain) + noise_std*randn(size(Xtrain));


%% ============================================================
% 2. Train Gaussian Process
% =============================================================

GP = fitrgp(Xtrain, Ytrain, ...
    'KernelFunction','squaredexponential', ...
    'BasisFunction','constant', ...
    'Standardize',true, ...
    'FitMethod','exact', ...
    'PredictMethod','exact');


%% ============================================================
% 3. Prediction grid
% =============================================================

X = linspace(-3.3, 3.3, 500)';

% GP prediction
%
% IMPORTANT:
% In your MATLAB version, predict() returns:
%       mu    = GP mean
%       sigma = GP standard deviation
%
[mu, sigma] = predict(GP, X);

% True function
Ytrue = f(X);

% 95% GP uncertainty interval
upper = mu + 2*sigma;
lower = mu - 2*sigma;


%% ============================================================
% 4. Define OOD region
% =============================================================

OOD_boundary = 1.5;

% OOD region
OOD = X >= OOD_boundary;


%% ============================================================
% 5. Plot
% =============================================================

figure('Color','w', ...
       'Position',[100 100 950 520]);

hold on;


%% ------------------------------------------------------------
% OOD background
% ------------------------------------------------------------

yl = [-1.5 1.8];

patch([OOD_boundary 3.3 3.3 OOD_boundary], ...
      [yl(1) yl(1) yl(2) yl(2)], ...
      [1.0 0.90 0.90], ...
      'EdgeColor','none', ...
      'FaceAlpha',0.55);


%% ------------------------------------------------------------
% GP uncertainty band
% ------------------------------------------------------------

fill([X; flipud(X)], ...
     [upper; flipud(lower)], ...
     [0.65 0.80 0.95], ...
     'EdgeColor','none', ...
     'FaceAlpha',0.65);


%% ------------------------------------------------------------
% True function
% ------------------------------------------------------------

plot(X, Ytrue, ...
     'b-', ...
     'LineWidth',1.8);


%% ------------------------------------------------------------
% GP mean prediction
% ------------------------------------------------------------

plot(X, mu, ...
     '--', ...
     'Color',[0.95 0.45 0.05], ...
     'LineWidth',1.8);


%% ------------------------------------------------------------
% Training data
% ------------------------------------------------------------

plot(Xtrain, Ytrain, ...
     'ko', ...
     'MarkerFaceColor','k', ...
     'MarkerSize',4);


%% ------------------------------------------------------------
% OOD boundary
% ------------------------------------------------------------

xline(OOD_boundary, ...
      'r--', ...
      'LineWidth',1.5);


%% ============================================================
% 6. Labels and annotations
% =============================================================

xlabel('$x$', ...
       'Interpreter','latex', ...
       'FontSize',15);

ylabel('$f(x)$', ...
       'Interpreter','latex', ...
       'FontSize',15);


%% OOD label

text(2.35, 1.55, ...
     'OOD region', ...
     'Color',[0.85 0.05 0.05], ...
     'FontWeight','bold', ...
     'FontSize',12, ...
     'HorizontalAlignment','center');


%% High epistemic uncertainty label

text(2.35, -1.15, ...
     'High epistemic uncertainty', ...
     'Color',[0.85 0.05 0.05], ...
     'FontSize',11, ...
     'HorizontalAlignment','center');


%% Data-rich region label

text(-0.8, -1.05, ...
     {'Data-rich region', ...
      '(low epistemic uncertainty)'}, ...
     'Color',[0.00 0.35 0.85], ...
     'FontWeight','bold', ...
     'FontSize',11, ...
     'HorizontalAlignment','center');


%% ============================================================
% 7. Legend
% =============================================================

legend({'OOD region', ...
        'GP uncertainty ($\pm2\sigma$)', ...
        'True function', ...
        'GP mean (prediction)', ...
        'Training data'}, ...
       'Interpreter','latex', ...
       'Location','northwest', ...
       'FontSize',11);


%% ============================================================
% 8. Figure formatting
% =============================================================

xlim([-3.3 3.3]);
ylim(yl);

set(gca, ...
    'FontSize',13, ...
    'LineWidth',1.0, ...
    'Box','on');

grid off;

hold off;