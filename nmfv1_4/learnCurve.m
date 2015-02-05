function [besta,bestalpha,bestb,bestMin]=learnCurve(es,ns,option)
% learn the learning curve
% es: vector, the error rates
% ns: vector, the correponding numbers of trining samples
% option: struct, reserved, not used now
% example:
% es=[0.4;0.28;0.17;0.04;0.01;0.005;0.0005;0.0005];
% ns=[10; 20;  30; 40;50;60;80;100];
% [besta,bestalpha,bestb,bestMin]=learnCurve(es,ns)
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

if nargin<3
   option=[]; 
end
optionDefault.someField=[];
option=mergeOption(option,optionDefault);


f=@(x) invPowerLaw(x,ns,es);
x0=[0;1;0.01];
lb=[-100;0;0];
ub=[100;10;1];
[x,resnorm] = lsqnonlin(f,x0,lb,ub);
besta=x(1);
bestalpha=x(2);
bestb=x(3);
bestMin=resnorm;

% 
% bs=(0.2:-option.bstep:0)';
% numb=numel(bs);
% minRs=nan(numb,1);
% as=nan(numb,1);
% alphas=nan(numb,1);
% bestb=0;
% bestMin=inf;
% besta=0;
% bestalpha=0;
% for i=1:numb
% %     d=log(b-es);
% %     C=[-ones(numel(ns),1),ns];
% %     lb=[-100;0];
% %     ub=[];
% %     Aeq=[];
% %     beq=[];
% %     A=[];
% %     bb=[];
% %     [x,resnorm] = lsqlin(C,d,A,bb,Aeq,beq,lb,ub);
%     as(i)=exp(x(1));
%     alphas(i)=x(2);
%     minRs(i)=resnorm;
% end
% [bestMin,ind]=min(minRs);
% besta=as(ind);
% bestalpha=alphas(ind);
% bestb=bs(i);
end
