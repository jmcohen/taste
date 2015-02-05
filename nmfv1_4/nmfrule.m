function [A,Y,numIter,tElapsed,finalResidual]=nmfrule(X,k,option)
% NMF based on multiple update rules: X=AY, s.t. X,A,Y>=0.
% Definition:
%     [A,Y,numIter,tElapsed,finalResidual]=nmfrule(X,k)
%     [A,Y,numIter,tElapsed,finalResidual]=nmfrule(X,k,option)
% X: non-negative matrix, dataset to factorize, each column is a sample, and each row is a feature.
% k: number of clusters.
% option: struct:
% option.distance: distance used in the objective function. It could be
%    'ls': the Euclidean distance (defalut),
%    'kl': KL divergence.
% option.iter: max number of interations. The default is 1000.
% option.dis: boolen scalar, It could be 
%     false: not display information,
%     true: display (default).
% option.residual: the threshold of the fitting residual to terminate. 
%    If the ||X-XfitThis||<=option.residual, then halt. The default is 1e-4.
% option.tof: if ||XfitPrevious-XfitThis||<=option.tof, then halt. The default is 1e-4.
% A: matrix, the basis matrix.
% Y: matrix, the coefficient matrix.
% numIter: scalar, the number of iterations.
% tElapsed: scalar, the computing time used.
% finalResidual: scalar, the fitting residual.
% References:
%  [1]\bibitem{NMF_Lee1999}
%     D.D. Lee and S. Seung,
%     ``Learning the parts of objects by non-negative matrix factorization,''
%     {\it Science},
%     vol. 401, pp. 788-791, 1999.
%  [2]\bibitem{NMF_Lee2001}
%     D.D. Lee and S. Seung,
%     ``Algorithms for non-negative matrix factorization,''
%     {\it Advances in Neural Information Processing Systems}, 
%     vol. 13, pp. 556-562, 2001.
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
% May 01, 2011
%%%%

tStart=tic;
optionDefault.distance='ls';
optionDefault.iter=1000;
optionDefault.dis=true;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if nargin<3
   option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

% iter: number of iterations
[r,c]=size(X); % c is # of samples, r is # of features
Y=rand(k,c);
% Y(Y<eps)=0;
Y=max(Y,eps);
A=X/Y;
% A(A<eps)=0;
A=max(A,eps);
XfitPrevious=Inf;
for i=1:option.iter
    switch option.distance
        case 'ls'
            A=A.*((X*Y')./(A*(Y*Y')));
%             A(A<eps)=0;
                A=max(A,eps);
            Y=Y.*((A'*X)./(A'*A*Y));
%             Y(Y<eps)=0;
                Y=max(Y,eps);
        case 'kl'
            A=A.*(((X./(A*Y))*Y')./(ones(r,1)*sum(Y,2)'));
            A=max(A,eps);
            Y=Y.*((A'*(X./(A*Y)))./(sum(A,1)'*ones(1,c)));
            Y=max(Y,eps);
        otherwise
            error('Please select the correct distance: option.distance=''ls''; or option.distance=''kl'';');
    end
    if mod(i,100)==0 || i==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(i),'th']);
        end
        XfitThis=A*Y;
        fitRes=matrixNorm(XfitPrevious-XfitThis);
        XfitPrevious=XfitThis;
        curRes=norm(X-XfitThis,'fro');
        if option.tof>=fitRes || option.residual>=curRes || i==option.iter
            s=sprintf('Mutiple update rules based NMF successes! \n # of iterations is %0.0d. \n The final residual is %0.4d.',i,curRes);
            disp(s);
            numIter=i;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end
