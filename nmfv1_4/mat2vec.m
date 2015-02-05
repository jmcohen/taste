% y = MAT2VEC(X)  Given an m x n matrix X, this produces the vector y of length
%   m*n that contains the columns of the matrix X, stacked below each other.
%
% See also vec2mat.

function y = mat2vec(X)
[m n] = size(X);
y = reshape(X,m*n,1);
end