function [K, Pset] = l1NNQPActiveSet(CtC, CtA,lambda)
% kernel l1 Regularized NNQP solving min K^t CtC K + CtAK + lambda [||K_1||_1,...], s.t. K>=0, for given A and C.
% CtC: kernel matrix, k(C,C), Hessian.
% CtA: kernel matrix, k(C,A), Gradient.
% lambda: lambda.
% References:
% [1] \bibitem{NNLS2004}
% M.H. Van Benthem and M.R. Keenan,
% ``Fast algorithm for the solution of large-scale non-negaive constrained least squares problems,''
% {\it Journal of Chemometrics},
% vol. 18, pp. 441-450, 2004.
% [2]\bibitem{cibcb2012}
%     Y. Li and A. Ngom,
%     ``A New Kernel Non-Negative Matrix Factorization and Its Application in Microarray Data Analysis,''
%     {\it CIBCB},
%     submited, 2012.
% Yifeng Li, Dec. 30, 2011
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% Note: This function is modified from the code of Ref. [1].

% Check the input arguments for consistency and initialize
error(nargchk(2,3,nargin))
lVar = size(CtC,1);
pRHS = size(CtA,2);
W = zeros(lVar, pRHS);
iter=0; maxiter=3*lVar;

%CtC=(CtC+CtC')./2;

CtA=CtA+lambda;
% Obtain the initial feasible solution and corresponding passive set
% K = cssls(CtC, CtA);
% Pset = K > 0;
% K(~Pset) = 0;
% D = K;
% Fset = find(~all(Pset));

K=zeros(size(CtA));
Pset=K>0;
W = -CtA;
D = K;
Fset = find(~all(Pset));

   % Active set algorithm for NNLS main loop
oitr=0; % HKim
while ~isempty(Fset)
    
    oitr=oitr+1; %if oitr > 5, fprintf('%d ',oitr);, end % HKim
    
    % Solve for the passive variables (uses subroutine below)
    K(:,Fset) = cssls(CtC, CtA(:,Fset), Pset(:,Fset));
    % Find any infeasible solutions
    Hset = Fset(find(any(K(:,Fset) < 0)));
    % Make infeasible solutions feasible (standard NNLS inner loop)
    if ~isempty(Hset)
      nHset = length(Hset);
      alpha = zeros(lVar, nHset);
      while ~isempty(Hset) && (iter < maxiter)
            iter = iter + 1; 
            alpha(:,1:nHset) = Inf;
            % Find indices of negative variables in passive set
            [i, j] = find(Pset(:,Hset) & (K(:,Hset) < 0));
            if isempty(i), break, end
            hIdx = sub2ind([lVar nHset], i, j);
            if nHset==1, % HKim
                negIdx = sub2ind(size(K), i, Hset*ones(length(j),1)); %HKim
            else % HKim
               negIdx = sub2ind(size(K), i, Hset(j)');
            end % HKim
            alpha(hIdx) = D(negIdx)./(D(negIdx) - K(negIdx));
            [alphaMin,minIdx] = min(alpha(:,1:nHset));
            alpha(:,1:nHset) = repmat(alphaMin, lVar, 1);
            D(:,Hset) = D(:,Hset)-alpha(:,1:nHset).*(D(:,Hset)-K(:,Hset));
            idx2zero = sub2ind(size(D), minIdx, Hset);
            D(idx2zero) = 0;
            Pset(idx2zero) = 0;
            K(:, Hset) = cssls(CtC, CtA(:,Hset), Pset(:,Hset));
            Hset = find(any(K < 0)); nHset = length(Hset);
      end
   end%if
   % Make sure the solution has converged
   %if iter == maxiter, error('Maximum number iterations exceeded'), end
   % Check solutions for optimality
   W(:,Fset) = -CtA(:,Fset)-CtC*K(:,Fset);
   Jset = find(all(~Pset(:,Fset).*W(:,Fset) <= 0));
   Fset = setdiff(Fset, Fset(Jset));
   % For non-optimal solutions, add the appropriate variable to Pset
   if ~isempty(Fset)
       [mx, mxidx] = max(~Pset(:,Fset).*W(:,Fset));
       Pset(sub2ind([lVar pRHS], mxidx, Fset)) = 1;
       D(:,Fset) = K(:,Fset);
   end
end
end
% ****************************** Subroutine****************************
function [K] = cssls(CtC, CtA, Pset)
% Solve the set of equations CtA = CtC*K for the variables in set Pset
% using the fast combinatorial approach
tol=0;%2^(-32);
K = zeros(size(CtA));
if (nargin == 2) || isempty(Pset) || all(Pset(:))
    K = (CtC+tol*eye(size(CtC)))\(-CtA);
%     K = CtC\CtA;
%     K=pinv(CtC)*CtA;
%       K=invsvd(CtC)*CtA;
else
%    [lVar pRHS] = size(Pset);
%    codedPset = 2.^(lVar-1:-1:0)*Pset;
%    [sortedPset, sortedEset] = sort(codedPset);
%    breaks = diff(sortedPset);
%    breakIdx = [0 find(breaks) pRHS];
%    for k = 1:length(breakIdx)-1
%      cols2solve = sortedEset(breakIdx(k)+1:breakIdx(k+1));
%      vars = Pset(:,sortedEset(breakIdx(k)+1));
%      K(vars,cols2solve) = (CtC(vars,vars)+tol*eye(sum(vars),sum(vars)))\(-CtA(vars,cols2solve));
% %      K(vars,cols2solve) = CtC(vars,vars)\CtA(vars,cols2solve);
% %      K(vars,cols2solve) = pinv(CtC(vars,vars))*CtA(vars,cols2solve);
% %      K(vars,cols2solve) = invsvd(CtC(vars,vars))*CtA(vars,cols2solve);
%    end
  
   [lVar pRHS] = size(Pset);
%    [sortedPset, sortedEset] = sort(codedPset);
   codedPset=genCode(Pset);
   codeUnik=unique(codedPset);
   numUnik=numel(codeUnik);
   for k = 1:numUnik
     cols2solve = (codedPset==codeUnik(k));
     for i=1:pRHS
        if cols2solve(i)
           aCol=i;
           break;
        end
     end
     vars = Pset(:,aCol);
     K(vars,cols2solve) = (CtC(vars,vars)+tol*eye(sum(vars),sum(vars)))\(-CtA(vars,cols2solve));
%      K(vars,cols2solve) = CtC(vars,vars)\CtA(vars,cols2solve);
%      K(vars,cols2solve) = pinv(CtC(vars,vars))*CtA(vars,cols2solve);
%      K(vars,cols2solve) = invsvd(CtC(vars,vars))*CtA(vars,cols2solve);
   end
   
end
end
