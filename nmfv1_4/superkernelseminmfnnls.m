function [AtA,Y,numIter,tElapsed,finalResidual]=superkernelseminmfnnls(X,L,k,option)
% Supervised Kernel semi-NMF based on NNLS: phi(X)=A_phiY, s.t. Y>0, diag(AtA)=I.
% solve the problem:
% min_{A,Y,xi} 1/2|X-AY|_F^2 + 1/2|w|_2^2 + C'xi
% s.t. Z'w + xi <= L 
%      Y>=0
%      xi>=0
% where Z=[1;Y];
% to be finished...
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
% Aug 09, 2012
%%%%

tStart=tic;
optionDefault.C=1;
optionDefault.kernel='rbf';
optionDefault.param=[];
optionDefault.iter=200;
optionDefault.dis=1;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if nargin<4
    option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

% initialize
XtX=computeKernelMatrix(X,X,option);

if size(X,3)>1
    [r1,r2,c]=size(X); % c is # of samples, r is # of features
    r=r1*r2;
else
    [r,c]=size(X); % c is # of samples, r is # of features
end
% inx=kmeans(X',k); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
% H=(inx*ones(1,k)-ones(c,1)*cumsum(ones(1,k)))==0; % obtain logical matrix [1,0,0;1,0,0;0,1,0;0,1,0;1,0,0;0,0,1;...]
% Y=H'+0.2;
% use NMF to initialize Y
if ~strcmp(option.kernel,'linear')
    optionIni=option;
    optionIni.kernel='linear';
    [~,Y]=kernelseminmfnnls(X,k,optionIni);
else
    Y=rand(k,c);
end
% initialize AtA
if r==k
    AtA=Y'\XtX/Y;
    AtX=Y'\XtX;
else
    pinvY=pinv(Y);
    AtA=pinvY'*XtX*pinvY;
    AtX=pinvY'*XtX;
end
% normalize AtA and AtX
[AtA,~,XtX,~,AtX]=normalizeKernelMatrix(AtA,XtX,AtX);

% Y=rand(k,c);
prevRes=Inf;
% iterative updating
for i=1:option.iter
    % update Y
    [zeroA,tf0]=iszero(sum(AtA));
    AtA(zeroA,:)=[];
    AtA(:,zeroA)=[];
    AtX(zeroA,:)=[];
    if tf0 && k>1
        k=k-1;
    end
    if k<1
       error('k<1, impossible!'); 
    end
    Y=constrainedQPActive(AtA,AtX,-w',-L');
    % update w
    H=1+Y'*Y;
    lb=zeros(c,1);
    ub=C*ones(c,1);
    optQP=optimset('LargeScale',on);
    optQP=optimset(optQP,'TolX',1e-6);
    % note: change the signs of L
    mu = quadprog(H,-L,[],[],[],[],lb,ub,optQP);
    w=[ones(1,c);Y]*mu;
    xi=
    % update kernel matrices
    if r==k
        AtA=Y'\XtX/Y;
        AtX=Y'\XtX;
    else
        pinvY=pinv(Y);
        AtA=pinvY'*XtX*pinvY;
        AtX=pinvY'*XtX;
    end
    % normalize AtA and AtX
    [AtA,~,XtX,~,AtX]=normalizeKernelMatrix(AtA,XtX,AtX);
    if mod(i,20)==0 || i==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(i),'th']);
        end
        curRes=trace(XtX-Y'*AtX-AtX'*Y+Y'*AtA*Y);
        fitRes=prevRes-curRes;
        prevRes=curRes;
        if option.tof>=fitRes || option.residual>=curRes || i==option.iter
            disp(['kernelseminmfnnls successes!, # of iterations is ',num2str(i),'. The final residual is ',num2str(curRes)]);
            numIter=i;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end
