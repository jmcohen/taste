function [F,pvalF,acceptF,rankCl,CD]=FriedmanTest(Error,alpha)
% Friedman test for comparing multiple classifiers over multiple data sets
% Error: matrix, Acc_ij is the error rate of classifier j over the ith data
% alpha: the significant level
% F: scalar, the value of F statistic
% pvalF: scalar, p-value
% acceptF: logical, accept the null hypothesis or not
% rankCl: column vector, the average ranks of all classifiers
% CD: scalar, the crucial difference
% 
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
% Nov. 29, 2012

% get ranks
[N,k]=size(Error);
rank=zeros(N,k);
for i=1:N
   rank(i,:)=getRank(Error(i,:)); 
end
rankCl=mean(rank,1);
CD=[];
% Friedman test
xi2=(12*N)/(k*k+k) * (sum(rankCl.*rankCl) - (k*(k+1)*(k+1))/4  );
F=((N-1)*xi2)/(N*(k-1) - xi2);
pvalF=fpdf(F,k-1,(k-1)*(N-1));

if pvalF>alpha
   acceptF=true;
   return;
end

acceptF=false;
% post-hoc test, if rejected
% Nemenyi test
% critical values for the two-tailed Nemenyi test, 2-20 classifiers, df=inf
qalphas=[2.5760    2.9133    3.1113    3.2527    3.3658    3.4507    3.5285    3.5921    3.6487    3.6982    3.7406    3.7830    3.8184    3.8537    3.8820    3.9174    3.9386    3.9669    3.9952;
         1.9601    2.3405    2.5668    2.7294    2.8496    2.9486    3.0335    3.1042    3.1608    3.2173    3.2668    3.3093    3.3517    3.3941    3.4295    3.4578    3.4860    3.5143    3.5426;
         1.645     2.052     2.291     2.459     2.589     2.693     2.780     2.855     2.920     NaN       NaN       NaN       NaN       NaN       NaN       NaN        NaN      NaN       NaN];
% qalphas=[1.960,2.343,2.569,2.728,2.850,2.949,3.031,3.102,3.164;
%          1.645,2.052,2.291,2.459,2.589,2.693,2.780,2.855,2.920];
pvals=[0.01;0.05;0.10];
CD=qalphas(pvals==alpha,k-1)*sqrt((k*k+k)/(6*N));
end
