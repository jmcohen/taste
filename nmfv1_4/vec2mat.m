% Y = VEC2MAT(x,m,n)  
%   Given a vector of length m*n, this produces the m x n matrix
%   Y such that x = mat2vec(Y).  In other words, x contains the columns of the
%   matrix Y, stacked below each other.
%
% See also mat2vec.

function X = vec2mat(y,m,n)
X = reshape(y,m,n);
end