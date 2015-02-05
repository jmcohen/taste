% This is an example of using NMF to find patterns in the core-MYC
% signature.
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
% Feb. 26, 2013
%%%%

clear
load('coreMYC.mat','DWT','DMutate','geneSym','timePoints');

D=[DWT,DMutate];

optionsnmf2.alpha2=0.01;%0.01;
optionsnmf2.alpha1=0;
optionsnmf2.lambda2=0;
optionsnmf2.lambda1=0.01;
optionsnmf2.t1=false;
optionsnmf2.t2=true;

k=2;
[A,Y]=vsmf(D',k,optionsnmf2);

% rank the basis vectors
mA=mean(A,1);
if (mA(1)<mA(2))
   A=A(:,[2,1]); 
end

hNMF=plot(timePoints,A(1:8,:),'-','LineWidth',2);
hold on
hNMF=plot(timePoints,A(9:16,:),'-.','LineWidth',2);
hold off
set(gca,'XLim',[0,48]);
xlabel('Time Point');
ylabel('Gene Expression Level');
Legend(gca,{'Rising(Wide-Type)','Falling(Wide-Type)','Flat(Mutant)','Flat(Mutant)'},'Location','NorthEast');
print(gcf,'-depsc','-r200','MYCFeb252013.eps');

