function [A,Y,numIter,tElapsed,finalResidual]=nmf(X,k,option)
% nmf using various algorithms
% Definition:
%     [A,Y,numIter,tElapsed,finalResidual]=nmf(X,k)
%     [A,Y,numIter,tElapsed,finalResidual]=nmf(X,k,option)
% X: nonnegative matrix, or matrix of mixed signs, dataset to factorize, each column is a sample, and each row is a feature.
% k: scalar, number of clusters.
% option: struct:
% option. algorithm: the factorization algorithm to use. It could be
%     'nmfnnls': (Default) type "help nmfnnls" in the Matlab command line for more information;
%     'nmfrule': type "help nmfnnls" in the Matlab command line for more information;
%     'sparsenmfnnls': type "help nmfnnls" in the Matlab command line for more information;
%     'seminmfnnls': type "help nmfnnls" in the Matlab command line for more information;
%     'seminmfrule': type "help nmfnnls" in the Matlab command line for more information;
%     'convexnmfrule': type "help nmfnnls" in the Matlab command line for more information;
%     'orthnmfrule': type "help nmfnnls" in the Matlab command line for more information;
%     'kernelnmfrule': type "help nmfnnls" in the Matlab command line for more information;
%     'sparsenmf2rule': type "help sparsenmf2rule" in the Matlab command line for more information;
%     'sparsenmf2nnqp': type "help sparsenmf2nnqp" in the Matlab command line for more information;
%     'vsmf': type "help vsmf" in the Matlab command line for more information;
%     The the name of your own algorithm.
% option. optionnmf: struct, the option input of specific algorithm. The default is []. 
%     Use the same command above for more information.
% A: matrix, the basis matrix. 
%     Note: if option.algorithm='kernelnmfrule', A is a row of cells of
%     length 2. A{1}=A, that is the weight matrix, and A{2}=Ak, that is the kernel matrix.
% Y: matrix, the coefficient matrix.
% numIter: scalar, the number of iterations.
% tElapsed: scalar, the computing time used.
% finalResidual: scalar, the fitting residual.
%%%%
% Copyright (C) <2012>  <Yifeng Li>
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 20, 2011
%%%%

if nargin<3
   option=[]; 
end
optionDefault.algorithm='nmfnnls';
optionDefault.optionnmf=[];
option=mergeOption(option,optionDefault);
% if not non-negative data
if any(any(X<0)) && (strcmp(option.algorithm,'nmfnnls')||strcmp(option.algorithm,'nmfrule')||strcmp(option.algorithm,'orthnmfrule'))
    option.algorithm='seminmfnnls';
end
if any(any(X<0)) && (strcmp(option.algorithm,'sparsenmfnnls'))
    option.algorithm='sparseseminmfnnls';
end
switch option.algorithm
    case 'nmfnnls'
        [A,Y,numIter,tElapsed,finalResidual]=nmfnnls(X,k,option.optionnmf);
    case 'nmfrule'
        [A,Y,numIter,tElapsed,finalResidual]=nmfrule(X,k,option.optionnmf);
    case 'sparsenmfnnls'
        [A,Y,numIter,tElapsed,finalResidual]=sparsenmfnnls(X,k,option.optionnmf);
    case 'sparsenmfNNQP'
        [A,Y,numIter,tElapsed,finalResidual]=sparseNMFNNQP(X,k,option.optionnmf);
    case 'knmf-l1nnqp'
        [AtA,Y,numIter,tElapsed,finalResidual,A]=kernelSparseNMFNNQP(X,k,option.optionnmf);
    case 'seminmfnnls'
        [A,Y,numIter,tElapsed,finalResidual]=seminmfnnls(X,k,option.optionnmf);
    case 'seminmfrule'
        [A,Y,numIter,tElapsed,finalResidual]=seminmfrule(X,k,option.optionnmf);
    case 'convexnmfrule'
        [A,Y,numIter,tElapsed,finalResidual]=convexnmfrule(X,k,option.optionnmf);
        A=X*A;
    case 'sparseseminmfnnls'
        [A,Y,numIter,tElapsed,finalResidual]=sparseseminmfnnls(X,k,option.optionnmf);
    case 'knmf-nnls'
        [AtA,Y,numIter,tElapsed,finalResidual]=kernelseminmfnnls(X,k,option.optionnmf);
        A=AtA;
    case 'knmf-dc'
        [A,Y,numIter,tElapsed,finalResidual]=kernelnmfdecom(X,k,option.optionnmf);
    case 'knmf-ur'
        option.optionnmf
        [AtA,Y,numIter,tElapsed,finalResidual]=kernelseminmfrule(X,k,option.optionnmf);
        A=AtA;
    case 'knmf-cv'
        [~,A,Y,numIter,tElapsed,finalResidual]=kernelconvexnmf(X,k,option.optionnmf);
%         [~,A,Y,numIter,tElapsed,finalResidual]=kernelnmfrule(X,k,option.optionnmf);
    case 'orthnmfrule'
        [A,S,Y,numIter,tElapsed,finalResidual]=orthnmfrule(X,k,option.optionnmf);
        A=A*S;
    case 'kernelnmfrule'
        [Ak,A,Y,numIter,tElapsed,finalResidual]=kernelnmfrule(X,k,option.optionnmf);
        A{1}=A;
        A{2}=Ak;
    case 'wnmfrule'
        [A,Y,numIter,tElapsed,finalResidual]=wnmfrule(X,k,option.optionnmf);
    case 'sparsenmf2rule'
        [A,Y,numIter,tElapsed,finalResidual]=sparsenmf2rule(X,k,option.optionnmf);
    case 'sparsenmf2nnqp'
        [A,Y,numIter,tElapsed,finalResidual]=sparsenmf2nnqp(X,k,option.optionnmf);
    case 'vsmf'
        [A,Y,numIter,tElapsed,finalResidual]=vsmf(X,k,option.optionnmf);
    otherwise
        eval(['[A,Y,numIter,tElapsed,finalResidual]=',option.algorithm,'(X,k,option.optionnmf);']);
end
end
