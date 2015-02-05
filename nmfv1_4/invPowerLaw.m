function F=invPowerLaw(x,ns,es)
% inverse power law 
% numN=numel(ns);

F=x(1)*(ns.^(-x(2)))+x(3) - es;
end