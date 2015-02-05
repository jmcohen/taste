function [P,Q,R,SS,Xhat]=estimatePQRS(Y,rank)
% Estimate the paramaters of the dynamical systems
% This function is called by ds kernel
% Y: matrix
% rank: scalar, matrix rank
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
%%%%

[numR,numC]=size(Y); % numbers of genes and time points
[U,S,V]=svd(Y);
U=U(:,1:rank); % numR by rank
S=S(1:rank,1:rank); % rank by rank
V=V(:,1:rank); % numC by rank
P=U;
Xhat=S(1:rank,1:rank)*V(:,1:rank)';

E1=eye(numC-1,numC-1);
E1=[zeros(1,numC-1);E1];
E1=[E1,zeros(numC,1)];
E2=eye(numC-1,numC-1);
E2=[E2;zeros(1,numC-1)];
E2=[E2,zeros(numC,1)];
sVec=diag(S);
sVecInv=1./sVec;
SInv=diag(sVecInv);
Q=S*V'*E1*V/(V'*E2*V)*SInv;

SS=zeros(rank,rank);
for t=1:numC-1
    vtPlus1=Xhat(:,t+1)-Q*Xhat(:,t);
    SS=SS+vtPlus1*vtPlus1';
end
SS=SS/(numC-1);

R=zeros(numR,numR);
for t=1:numC
    wt=Y(:,t) - P*Xhat(:,t);
    R=R+wt*wt';
end
R=R/numC;
end