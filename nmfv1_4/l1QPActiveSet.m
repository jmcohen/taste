function [Xk,Uk]=l1QP(H,C,lambda)
% solve 0.5*tr(X^THX) + tr(CX) + lambda|X|_1
% Yifeng Li, Feb. 21, 2012

option.tol=2^(-32);

nV=size(H,1); % number of variables
nP=size(C,2);% number of problems
% Xk=H\(-C);%pinv(H)*(-C);
Xk=zeros(size(C));
Uk=abs(Xk);
Zk=[Xk;Uk];
W01=false(nV,nP);
W02=false(nV,nP);
W01(Xk>=0)=true;
W02(Xk<=0)=true;
Wk=[W01;W02];
A=[eye(nV),-eye(nV);-eye(nV),-eye(nV)]; % constraint
unsolved=true(nP,1);% indicators of unsolve problems
while any(unsolved)
    % compute pks
    Pk=EQP(H,C(:,unsolved),Xk(:,unsolved),Wk(:,unsolved),lambda);
    unSolvedNum=find(unsolved); % numerical index of unsolved problems
    Pk0=(sum(abs(Pk),1)<=option.tol); % logical index pk=0
    %there are some pk ==0
    if any(Pk0)
        Pk0Num=find(Pk0);% numerical index pk=0
        Xk0=Xk(:,unSolvedNum(Pk0Num));
        nPk0=numel(Pk0Num); % number pks who reach 0
        % compute Lagrange multiplier Lambda (different from lambda)
        Gk0=H*Xk0+C(:,unSolvedNum(Pk0Num));
        Gk0=-[Gk0;lambda.*ones(nV,nPk0)];
        Lambda=getLambda(A,Gk0,Wk(:,Pk0));
        [minLambda,minLambdaInd]=min(Lambda,[],1);
        LambdaLess0=(minLambda<=-option.tol);
        unsolved(unSolvedNum(Pk0Num(~LambdaLess0)))=false;  % record solved problems
        if ~any(unsolved)
            return; % optimization complete
        end
        % delet an active constraint from working set
        for i=1:nPk0
            if  ~isempty(LambdaLess0)&&LambdaLess0(i)
                Wk(minLambdaInd(i),unSolvedNum(Pk0Num(i)))=false; % delet this active constraint from active set
                wk=Wk(:,unSolvedNum(Pk0Num(i)));
                wkOr=(wk(1:nV)|wk(nV+1:end));
                while sum(wkOr)<nV % there exists wild variables
                    wildNum=find(wkOr==false); % numerical index of wild variable
                    wildNum=[wildNum(:);nV+wildNum(:)];
                    wk(wildNum(randi(numel(wildNum))))=true; % add one wild
                    wkOr=(wk(1:nV)|wk(nV+1:end));
                end
                Wk(:,unSolvedNum(Pk0Num(i)))=wk;
            end
        end
    end % if any(Pk0)
    
    % for pk~=0
    Pkn0=~Pk0;
    if any(Pkn0)
        Pkn0Num=find(Pkn0);
        nPkn0=numel(Pkn0Num);
        for i=1:nPkn0
            passive=~Wk(:,unSolvedNum(Pkn0Num(i))); % take passive set of current problem
            if ~any(passive) % passive set is empty
                continue;
            end
            Ap=A(passive,:);
            passiveNum=find(passive);
            Appk=Ap*Pk(:,Pkn0Num(i));
            Appkg0Num=find(Appk>option.tol); % numerical index that ai^Tpk>0
            if ~isempty(Appkg0Num)
            passiveNumg0=passiveNum(Appkg0Num);
            [alphak,alphakInd]=min(-(Ap(Appkg0Num,:)*Zk(:,unSolvedNum(Pkn0Num(i))))./(Appk(Appkg0Num)));
%             if alphak<0
%                 alphak=0;
%             end
            alphak1=min([1,alphak]); % minmum alpha
            else
                alphak1=1;
            end
            Zk(:,unSolvedNum(Pkn0Num(i)))=Zk(:,unSolvedNum(Pkn0Num(i)))+alphak1.*Pk(:,Pkn0Num(i)); % update solution
            Xk=Zk(1:nV,:);
            Uk=Zk(2*nV:end,:);
            if alphak1<1
                Wk(passiveNumg0(alphakInd),unSolvedNum(Pkn0Num(i)))=true;
            end
        end
    end
end
end