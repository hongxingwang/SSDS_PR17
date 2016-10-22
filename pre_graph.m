%=====================================================================
%   This function is to obtain edge weights and max degrees 
%   of similarity graph and dissimilarity graph for
%   Structured Sparse Dicationary Selection
%
%   Citation
%   --------
%   Hongxing Wang, Yoshinobu Kawahara, Chaoqun Weng, and Junsong Yuan,
%   "Representative Selection with Structured Sparsity",
%   Pattern Recognition, vol. 63, pp. 268-278, 2017,
%   http://dx.doi.org/10.1016/j.patcog.2016.10.014.
%=====================================================================
function [m,n,graph] = pre_graph(X,param)

%   INPUTS
%   ------
%   X: data matrix, m*n
%   param.k_lg = 3;   % for k ( = k_lg ) largest neighbors
%   param.k_sm = 3;   % for k ( = k_sm ) nearest neighbors
%   param.lambda_row; % regularization parameter for diversity 
%   param.lambda_col; % regularization parameter for locality-sensitivity 


%   OUTPUTS
%   -------
%   m:    # dim of data
%   n:    # data
%   graph.graph_col.weights:    edge weights of similarity graph
%   graph.graph_row.weights:    edge weights of dissimilarity graph
%   graph.graph_col.max_degree: max degree of similarity graph
%   graph.graph_row.max_degree: max degree of dissimilarity graph

%=====================================================================


[m,n] = size(X); %% n data of m-dim


%% edge weights of dissimilarity graph 
if param.lambda_row == 0
    W_row = sparse(n,n);
else 
    W_row = W_row_diversity_sqEuler(X,param.k_lg);
    W_row = param.lambda_row *W_row;
end


%% edge weights of similarity graph
if param.lambda_col == 0
    W_col = sparse(n,n);
else 
    W_col = W_col_sensitivity(X,param.k_sm);
    W_col = param.lambda_col * W_col;
end
%% converting weights into sparse form
W_col = sparse(W_col);
W_row = sparse(W_row);

graph.graph_col.weights = W_col;
graph.graph_row.weights = W_row;


%% max degree of each graph
graph.graph_col.max_degree = max(sum(graph.graph_col.weights));
graph.graph_row.max_degree = max(sum(graph.graph_row.weights));


function W_col = W_col_sensitivity(X,k_sm)
% edge weights of similarity graph
[~,n] = size(X); % X; n data of m-dim

W_col = zeros(n,n);
sqD = sqDistance(X, X);
sqSigma = median(reshape(sqD,1,n*n));
sqD = 0.5*sqD./sqSigma;
exp_sqD = exp(-sqD);

[sorted_sqD,~] = sort(sqD,1,'ascend');  % sorted based on column
selected_s = sqD <= sorted_sqD(k_sm,:)'*ones(1,n);
selected_s = selected_s.*( ones(n)-eye(n) );
selected_s = selected_s & selected_s';  % mutual k-nearest n
W_col(selected_s > 0) = exp_sqD(selected_s > 0);
W_col = sparse(W_col);

function W_row = W_row_diversity_sqEuler(X,k_lg)
% edge weights of dissimilarity graph
[~,n] = size(X); % X; n data of m-dim

W_row = zeros(n,n);
sqD = sqDistance(X, X);
sqD = sqD./max(sqD(:));

[sorted_sqD,~] = sort(sqD,1,'descend'); % sorted based on column
selected_l = sqD >= sorted_sqD(k_lg,:)'*ones(1,n);
selected_l = selected_l.*( ones(n)-eye(n) );
selected_l = selected_l & selected_l';  % mutual k-largest n

W_row(selected_l > 0) = ( sqD(selected_l > 0) );
W_row = sparse(W_row);

function D = sqDistance(X, Y)
% X d*n
% Y d*m
D = bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
