function [Xk]=constrainedQPActive(H,C,A,B)
% solve 0.5*tr(X^THX + CX)
% s.t. A*X -B <=0
%      -X<=0
% where A is a row vector, B is a row vector
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
% Feb. 21, 2012
%%%%


option.tol=2^(-32);

nV=size(H,1); % number of variables
nP=size(C,2);% number of problems
% compute feasiable points
Xk=zeros(nV,nP);%feasible(H,C,A,B);
W01=false(nV,nP);
W02=false(1,nP);
W01(Xk<=option.tol)=true;
W02(A*Xk-B>=-option.tol)=true;
Wk=[W01;W02];
%A=[eye(nV),-eye(nV);-eye(nV),-eye(nV)]; % constraint
unsolved=true(nP,1);
while any(unsolved)
    % compute pks
    Pk=EQP0(H,C(:,unsolved),Xk(:,unsolved),Wk(:,unsolved),A,B);
    unSolvedNum=find(unsolved); % numerical index of unsolved problems
    Pk0=(sum(abs(Pk),1)<=option.tol); % logical index pk=0
    %there are some pk ==0
    if any(Pk0)
        Pk0Num=find(Pk0);% numerical index pk=0
        Xk0=Xk(:,unSolvedNum(Pk0Num));
        nPk0=numel(Pk0Num); % number pks who reach 0
        % compute Lagrange multiplier Lambda (different from lambda)
        Gk0=H*Xk0+C(:,unSolvedNum(Pk0Num));
        Gk0=-Gk0;
        Lambda=getLambda(A,Gk0,Wk(:,Pk0));
        [minLambda,minLambdaInd]=min(Lambda,[],1);
        LambdaLess0=(minLambda<-option.tol);
        unsolved(unSolvedNum(Pk0Num(~LambdaLess0)))=false;  % record solved problems
        if ~any(unsolved)
            return; % optimization complete
        end
        % delet an active constraint from working set
        for i=1:nPk0
            if  ~isempty(LambdaLess0)&&LambdaLess0(i)
                Wk(minLambdaInd(i),unSolvedNum(Pk0Num(i)))=false; % delet this active constraint from active set
            end
        end
    end % if any(Pk0)
    
    % for pk~=0
    Pkn0=~Pk0;
    if any(Pkn0)
        Pkn0Num=find(Pkn0);
        nPkn0=numel(Pkn0Num);
        for i=1:nPkn0 % for each problem whose step is not zero vector
            passive=~Wk(:,unSolvedNum(Pkn0Num(i))); % take passive set of current problem
            if ~any(passive) % passive set is empty
                continue;
            end
            passiveNum=find(passive);
            if any(passiveNum==nV+1) && A*Pk(:,Pkn0Num(i))>option.tol % Ax-B<0 and Apk>0
                alpham=(B(unSolvedNum(Pkn0Num(i)))-A*Xk(:,unSolvedNum(Pkn0Num(i))))/(A*Pk(:,Pkn0Num(i)));
                passive(end)=[];
                passiveNum(passiveNum==nV+1)=[];
            else 
                alpham=1;
            end
            Appk=Pk(:,Pkn0Num(i));
            apxk=Xk(:,unSolvedNum(Pkn0Num(i)));
            apxk=apxk(Appk<-option.tol); % b_i - a_i'x_k
            Appk=-Appk(Appk<-option.tol); % a_i'p_k
            Appkg0Num=find(Appk>option.tol); % numerical index that ai^Tpk>0
            if ~isempty(Appkg0Num)
            passiveNumg0=passiveNum(Appkg0Num);
            [alphak,alphakInd]=min(apxk./(Appk(Appkg0Num)));
%             if alphak<0
%                 alphak=0;
%             end
            alphak1=min([1,alphak,alpham]); % minmum alpha
            else
                alphak1=1;
            end
            Xk(:,unSolvedNum(Pkn0Num(i)))=Xk(:,unSolvedNum(Pkn0Num(i)))+alphak1.*Pk(:,Pkn0Num(i)); % update solution
            if alphak1<1 % add to working set
                if alphak<=alpham
                    j=passiveNumg0(alphakInd);
                else
                    j=nV+1;
                end
                Wk(j,unSolvedNum(Pkn0Num(i)))=true;
            end
        end
    end
end
end