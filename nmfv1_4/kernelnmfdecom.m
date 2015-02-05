function [A,Y,numIter,tElapsed,finalResidual]=kernelnmfdecom(X,k,option)
% Kernel semi-NMF based on decomposing kernel matrix: XtX=AY, s.t. Y>0. 
% Definition:
%     [A,Y,numIter,tElapsed,finalResidual]=kernelnmfdecom(X,k)
%     [A,Y,numIter,tElapsed,finalResidual]=kernelnmfdecom(X,k,option)
% X: matrix of mixed signs, dataset to factorize, each column is a sample, and each row is a feature.
% k: scalar, number of basis vectors.
% option: struct:
% option.kernel: see function computeKernelMatrix
% option.param: see function computeKernelMatrix
% option.iter: max number of interations. The default is 1000.
% option.dis: boolen scalar, It could be 
%     false: not display information,
%     true: display (default).
% option.residual: the threshold of the fitting residual to terminate. 
%     If the ||X-XfitThis||<=option.residual, then halt. The default is 1e-4.
% option.tof: if ||XfitPrevious-XfitThis||<=option.tof, then halt. The default is 1e-4.
% A: matrix, the basis matrix
% Y: matrix, the coefficient matrix.
% numIter: scalar, the number of iterations.
% tElapsed: scalar, the computing time used.
% finalResidual: scalar, the fitting residual.
% References:
% [1] \bibitem{Zhang2006}
%  D. Zhang, Z. Zhou, S. Chen,
%  ``Non-negative Matrix Factorization on Kernels,''
%  {\it LNCS},
%  vol. 4099, pp. 404-412, 2006.
% [2]\bibitem{cibcb2012}
%     Y. Li and A. Ngom,
%     ``A New Kernel Non-Negative Matrix Factorization and Its Application in Microarray Data Analysis,''
%     {\it CIBCB},
%     pp. 371-378, 2012.
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
% Sep 27, 2011
%%%%

tStart=tic;
optionDefault.kernel='rbf';
optionDefault.param=[];
optionDefault.iter=1000;
optionDefault.dis=1;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if nargin<3
   option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

% initialize
XtX=computeKernelMatrix(X,X,option);
[A,Y,numIter,tElapsed,finalResidual]=nmfrule(XtX,k,option);
tElapsed=toc(tStart);
end
