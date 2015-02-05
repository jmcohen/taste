% This is an example of how to draw learning curves
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
%%%%

clear

% error rates of NNLS
es=   [0.2393;    0.0983;    0.0735;    0.0579;    0.0400;    0.0235;    0.0246;    0.0067]; % mean
e25s=[0.1754;    0.0417;    0.0465;    0.0263;    0.0303;         0;         0;         0]; % 25th quantile
e75s=[0.2982;    0.1250;    0.0930;    0.0789;    0.0606;    0.0435;    0.0769;         0]; % 75th quantile
numTrains=[6;     15;        20;        25;        30;        40;        50;        60];

limit=[0,200,0,0.5];
[besta,bestalpha,bestb,bestMin]=learnCurve(es,numTrains)
a=besta;
alpha=bestalpha;
b=bestb;
f=@(x) a*x^(-alpha)+b;
fplot(f,limit,'b-');
% ezplot(f,[0,500]);
fprintf('NNLS Mean Parameter: (%.2f,%.2f,%.2f)\n',a,alpha,b);


[besta25,bestalpha25,bestb25,bestMin25]=learnCurve(e25s,numTrains)
a=besta25;
alpha=bestalpha25;
b=bestb25;
f=@(x) a*x^(-alpha)+b;
hold on
fplot(f,limit,'b-.');
% ezplot(f,[0,500]);
fprintf('NNLS 25 Parameter: (%.2f,%.2f,%.2f)\n',a,alpha,b);

[besta75,bestalpha75,bestb75,bestMin75]=learnCurve(e75s,numTrains)
a=besta75;
alpha=bestalpha75;
b=bestb75;
f=@(x) a*x^(-alpha)+b;
hold on
fplot(f,limit,'b--');
% ezplot(f,[0,500]);
fprintf('NNLS 75 Parameter: (%.2f,%.2f,%.2f)\n',a,alpha,b);
hold on

% error rates of SVM
es=   [0.4877;    0.4567;    0.2884;    0.1945;    0.0496;    0.0292;    0.0800]; % mean
e25s=[0.4318;    0.4186;    0.2368;    0.1515;         0;         0;         0]; % 25th quantile
e75s=[0.5227;    0.4884;    0.3158;    0.2121;    0.0870;    0.0769;         0]; % 75th quantile
numTrains=[19;    20;        25;        30;        40;        50;        60];


[besta,bestalpha,bestb,bestMin]=learnCurve(es,numTrains)
a=besta;
alpha=bestalpha;
b=bestb;
f=@(x) a*x^(-alpha)+b;
fplot(f,limit,'r-');
% ezplot(f,[0,500]);
fprintf('SVM Mean Parameter: (%.2f,%.2f,%.2f)\n',a,alpha,b);

[besta25,bestalpha25,bestb25,bestMin25]=learnCurve(e25s,numTrains)
a=besta25;
alpha=bestalpha25;
b=bestb25;
f=@(x) a*x^(-alpha)+b;
hold on
fplot(f,limit,'r-.');
% ezplot(f,[0,500]);
fprintf('SVM 25 Parameter: (%.2f,%.2f,%.2f)\n',a,alpha,b);

[besta75,bestalpha75,bestb75,bestMin75]=learnCurve(e75s,numTrains)
a=besta75;
alpha=bestalpha75;
b=bestb75;
f=@(x) a*x^(-alpha)+b;
hold on
fplot(f,limit,'r--');
% ezplot(f,[0,500]);
fprintf('SVM 75 Parameter: (%.2f,%.2f,%.2f)\n',a,alpha,b);

xlabel('Sample Size','FontSize',14);
ylabel('Error Rate','FontSize',14);
set(gca,'FontSize',14);

legend({'NNLS Average','NNLS 25% Quantile','NNLS 75 Quantitle','SVM Mean','SVM 25% Quantitle','SVM 75% Quantitle'});
hold off
print(gcf,'-depsc2','-r300','SRBCTLearnCurve.eps');
