function [A,Y,numIter,tElapsed,finalResidual]=sparsenmfnnls(X,k,option)
% Sparse-NMF based on NNLS: X=AY, s.t. X,A,Y>0.
% Definition:
%     [A,Y,numIter,tElapsed,finalResidual]=sparsenmfnnls(X,k)
%     [A,Y,numIter,tElapsed,finalResidual]=sparsenmfnnls(X,k,option)
% X: non-negative matrix, dataset to factorize, each column is a sample, and each row is a feature.
% k: scalar, number of clusters.
% option: struct:
% option.eta: scalar, eta>0, the parameter to suppress A. The default is (max(max(X)))^2.
% option.beta: scalar, beta>0, the parameter to enforce sparsity on Y. 
%     The greater beta is, the sparser Y is. The default is 0.1.
%     If option.eta=0 and option.beta=0, it is equivalent to nmfnnls.
% option.iter: max number of interations. The default is 1000.
% option.dis: boolen scalar, It could be 
%     false: not display information,
%     true: display (default).
% option.residual: the threshold of the fitting residual to terminate. 
%     If the ||X-XfitThis||<=option.residual, then halt. The default is 1e-4.
% option.tof: if ||XfitPrevious-XfitThis||<=option.tof, then halt. The default is 1e-4.
% A: matrix, the basis matrix.
% Y: matrix, the coefficient matrix.
% numIter: scalar, the number of iterations.
% tElapsed: scalar, the computing time used.
% finalResidual: scalar, the fitting residual.
% Reference:
%  [1]\bibitem{NMF_ANLS_Kim2008}
%     H. Kim and H. Park,
%     ``Nonnegative matrix factorization based on alternating nonnegativity constrained least squares and active set method,''
%     {\it SIAM J. on Matrix Analysis and Applications},
%     vol. 30, no. 2, pp. 713-730, 2008.
%  [2]\bibitem{NMF_Sparse_Kim2007}
%     H. Kim and H. Park,
%     ``Sparse non-negatice matrix factorization via alternating non-negative-constrained least squares for microarray data analysis,''
%     {\it Bioinformatics},
%     vol. 23, no. 12, pp. 1495-1502, 2007.
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
% May 03, 2011
%%%%

tStart=tic;
optionDefault.eta=(max(max(X)))^2;
optionDefault.beta=0.1;
optionDefault.iter=200;
optionDefault.dis=true;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if nargin<3
   option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

if option.eta==0 || option.beta==0
    tElapsed=toc(tStart);
    [A,Y,numIter,tElapsed,finalResidual]=nmfnnls(X,k,option);
    return;
end

[r,c]=size(X); % c is # of samples, r is # of features
A=rand(r,k);
A=normc(A);
% Y=rand(k,c);
% Y=normc(Y);
XfitPrevious=Inf;
for i=1:option.iter
    Ae=[A;sqrt(option.beta)*ones(1,k)];
    X0=[X;zeros(1,c)];
    Y=kfcnnls(Ae,X0);
    YIt=[Y';sqrt(option.eta)*eye(k,k)];
    X0t=[X';zeros(k,r)];
    A=kfcnnls(YIt,X0t);
    A=A';
    if mod(i,20)==0 || i==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(i),'th']);
        end
        XfitThis=A*Y;
        fitRes=matrixNorm(XfitPrevious-XfitThis);
        XfitPrevious=XfitThis;
        curRes=norm(X-XfitThis,'fro');
        if option.tof>=fitRes || option.residual>=curRes || i==option.iter
            s=sprintf('NNLS based Sparse NMF successes! \n # of iterations is %0.0d. \n The final residual is %0.4d.',i,curRes);
            disp(s);
            numIter=i;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end

