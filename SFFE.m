function [Z, W, b] = SFFE(Xl,Xu,labels,L,Alpha,Mu,Gamma)
% SFFE: Semi-supervised Flexible Feature Extraction
%
% 
%         Input:
%           Xl         - Labeled Matrix (each column represent a sample).
%           Xu         - Unlabeled Matrix (each column represent a sample).
%         labels       - Label vector containing the labels of Xl matrix.
%           L          - Laplacian Matrix.
%     Alpha,Mu&Gamma   - Balance parameters.
%
% 
% 
% 
% 
%         Output:
%           Z          - Matrix of embeddings. (each row represent the embedding of its correspomding sample).
%           W          - Transformation matrix.
%           b          - Bias term.


X=[Xl,Xu];
[Dl, Ml] = MlMatrix(Xl,labels);
ExtMl = zeros(size(X,2));
ExtMl(1:size(Ml,2),1:size(Ml,2)) = Ml;
InvExtDl = zeros(size(X,2));
InvExtDl(1:size(Dl,2),1:size(Dl,2)) = diag(diag(Dl).^-1);
InvExtDl(size(Dl,2)+1:end, size(Dl,2)+1:end) = 1000 * eye(size(Xu,2));
L1 = L + Alpha * ExtMl;

D = size(X,1);
m = size(X,2);
one = ones(m,1);
Im = eye(m);
Hc = Im - 1/m * (one*one');
Xc = X * Hc;
N = (Xc' * Xc) / (Gamma * (Xc' * Xc) + Im);

[Z, eigVal] = eig(InvExtDl * (L1 + Mu * Gamma * Hc - Mu * (Gamma^2)*N));

[~, idx ] = sort(real(diag(eigVal)) ,'ascend');
Z = real(Z(:,idx));

A = Gamma * (Gamma * X*Hc*X' + eye(D)) \ Xc;
W = A * Z;
b = 1/m * (Z' * one - W' * X * one);

end