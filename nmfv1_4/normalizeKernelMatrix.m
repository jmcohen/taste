function [nAtA,normA,nBtB,normB,nAtB]=normalizeKernelMatrix(AtA,BtB,AtB)
% normalize random kernel
% Yifeng Li
% Feb. 08, 2012

nAtA=[];
normA=[];
nBtB=[];
normB=[];
nAtB=[];

normA=sqrt(diag(AtA));
nAtA=AtA./(normA*normA');

if nargin==1
   return; 
end

normB=sqrt(diag(BtB));
nBtB=BtB./(normB*normB');

if nargin==2
   return; 
end

nAtB=AtB./(normA*normB');
end
