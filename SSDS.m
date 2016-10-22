%=====================================================================
%REPRESENTATIVE SELECTION WITH STRUCTURED SPARSITY
%
%   Citation
%   --------
%   Hongxing Wang, Yoshinobu Kawahara, Chaoqun Weng, and Junsong Yuan,
%   "Representative Selection with Structured Sparsity",
%   Pattern Recognition, vol. 63, pp. 268-278, 2017,
%   http://dx.doi.org/10.1016/j.patcog.2016.10.014.
%
%   Environment
%   -----------
%   Matlab R2014b or later
%=====================================================================
function [sMat] = SSDS(X, param, opts)

%   INPUTS
%   ------
%   X: data matrix, #dim * #data
%   param.k_lg = 3;   % for k ( = k_lg ) largest neighbors
%   param.k_sm = 3;   % for k ( = k_sm ) nearest neighbors
%   param.alpha_l1l2; % hyper-parameter for row sparsity
%   param.alpha_row;  % hyper-parameter for diversity
%   param.alpha_col;  % hyper-parameter for locality-sensitivity

%   opts.maxIter_FISTA:    # max iterations for FISTA
%   opts.maxIter_Dykatra:  # max iterations for Dykatra
%   opts.thr_Dykatra:      tolerance between Dykatra iterations
%   opts.thr_FISTA:        tolerance between FISTA iterations

%   OUTPUTS
%   -------
%   sMat: selection matrix, #data * #data
%=====================================================================
% Lipschitz constant of d||X-XS||_F^2/dS
XTX = X'*X; LipC = norm(XTX,'fro'); 
% setting penalty parameters for proximal operators
% by \hat(lambda_i) = (cal_lambda/LipC)/alpha_i
cal_lambda = computeLambda_mat(X,true);
coeff = cal_lambda/LipC;
if ~isfield(param,'lambda_l1l2') 
    param.lambda_l1l2 = coeff/param.alpha_l1l2;
end
if ~isfield(param,'lambda_row')
    param.lambda_row = coeff/param.alpha_row;
end
if ~isfield(param,'lambda_col')
    param.lambda_col = coeff/param.alpha_col;
end

% getting graphs, esp. the weights
param_garph = struct('lambda_row', param.lambda_row, 'lambda_col', ...
    param.lambda_col, 'k_lg', param.k_lg, 'k_sm', param.k_sm);
[~,~,graph] = pre_graph(X,param_garph);

% finding selection matrix via Structured Sparse Dicationary Selection
tols_FT.ITER = opts.maxIter_FISTA;
tols_FT.ERR = opts.thr_FISTA;
tols_DY.ITER = opts.maxIter_Dykatra;
tols_DY.ERR = opts.thr_Dykatra;
sMat = fista_dykastra(param.lambda_l1l2,1/LipC,XTX,graph.graph_row,...
    graph.graph_col,tols_FT,tols_DY);
pause(0.1) % keep the pause
clear mex

