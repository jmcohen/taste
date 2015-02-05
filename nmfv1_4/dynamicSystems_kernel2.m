function K=dynamicSystems_kernel2(D1,D2,param)
% Dynamical system kernel function
% D1,D2: 3-way tensors, each frontal slice is a sample
% param: row vector, [rank,lambda]
% K: matrix, the kernel matrix
% Reference:
% K.M. Borgwardt, Class prediction from time series gene expression
% profiles using dynamical systems kernels. Pracific Symposium on
% Biocomputing, vol. 11, pp. 547-558, 2006.
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

if nargin<3
    param=[1,5];
end
if isempty(param)
   param=[1,5]; 
end
% n-way normalization
% Cent=[0,0,1];
% Scal=[1,1,0];
% [D1,Means,Scales]=nprocess(D1,Cent,Scal);
% [D2,Means,Scales]=nprocess(D2,Cent,Scal,Means,Scales,1);

% normalize each sample
% numR=size(D1,1);
% numC=size(D1,2);
% D1=matrizicing(D1,3);
% D2=matrizicing(D2,3);
% D1=D1';
% D2=D2';

% mean0 std 1
% [D1,trainSetMean,trainSetSTD]=normmean0std1(D1');
% D1=D1';
% D2=normmean0std1(D2',trainSetMean,trainSetSTD);
% D2=D2';

% unitnorm2
% D1=normc(D1);
% D2=normc(D2);

% D1=D1';
% D2=D2';
% numP1=size(D1,1); % number of pages
% numP2=size(D2,1); % number of pages
% D1=unmatrizicing(D1,3,[numR,numC,numP1]);
% D2=unmatrizicing(D2,3,[numR,numC,numP2]);

rank=param(1);
lambda=param(2);
numP1=size(D1,3); % number of pages
numP2=size(D2,3); % number of pages
K=nan(numP1,numP2);

for p1=1:numP1
    % estimate the parameters (P,Q,R,SS) of sample p1
   [P1,Q1,R1,S1,X1]=estimatePQRS(D1(:,:,p1),rank);   
    for p2=1:numP2
         % estimate the parameters (P,Q,R,S) of sample p2
       [P2,Q2,R2,S2,X2]=estimatePQRS(D2(:,:,p2),rank);
         % calculate M1 and M2
         M1=dlyap(exp(-lambda)*Q1',Q2',exp(-lambda)*Q1'*P1'*P2*Q2);
         M2=dlyap(exp(-lambda)*Q1',Q2',P1'*P2);
         %calculate the similarity of samples p1 and p2
        K(p1,p2)=X1(:,1)'*M1*X2(:,1) + (1/(exp(lambda) -1))*(trace(S1*M2)+trace(R1));
%         if p2<numP1
%            K(p2,p1)=K(p1,p2); 
%         end
    end
end
end