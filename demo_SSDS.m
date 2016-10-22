%=====================================================================
%   This is the demo code to run for finding the representatives via
%   Structured Sparse Dictionary Selection
%
%   Citation
%   --------
%   Hongxing Wang, Yoshinobu Kawahara, Chaoqun Weng, and Junsong Yuan,
%   "Representative Selection with Structured Sparsity",
%   Pattern Recognition, vol. 63, pp. 268-278, 2017,
%   http://dx.doi.org/10.1016/j.patcog.2016.10.014.
%=====================================================================
clear

if exist('fista_dykastra','file') ~= 3
    compile();
end


load('X','X') % X: data matrix, #dim * #data
n = size(X,2); 
Y = X - repmat(mean(X,2),1,n); % let the data be zero-mean


opts.maxIter_FISTA = 10;      % # max iterations for FISTA
opts.maxIter_Dykatra = 5;     % # max iterations for Dykatra
opts.thr_Dykatra = 1 * 10^-5; % tolerance between Dykatra iterations
opts.thr_FISTA = 1 * 10^-5;   % tolerance between FISTA iterations

param.k_lg = 3;           % for k ( = k_lg ) largest neighbors
param.k_sm = 3;           % for k ( = k_sm ) nearest neighbors
alpha = 10;
param.alpha_l1l2 = alpha; % hyper-parameter for row sparsity        
param.alpha_row  = alpha; % hyper-parameter for diversity
param.alpha_col  = alpha; % hyper-parameter for locality-sensitivity

% finding selection matrix via Structured Sparse Dicationary Selection
sMat = SSDS(Y, param, opts);

% finding indices of representatives
topk = 20;
q = 2; thrS = 0.99;
AdjSelMat = true;
if AdjSelMat
    sMat = sMat./(sMat'+eps);
end
sInd = findRep(sMat,thrS,q); % sample ranking & removal
repInd = sInd(1:topk);  % indices of top representatives

figure,plot(X(1,:),X(2,:),'k.'),
hold on,plot(X(1,repInd),X(2,repInd),'ro')
