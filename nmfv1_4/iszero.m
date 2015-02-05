function [tfFull,tf,n]=iszero(a)
% if vector a is zero vector then tf=true, if a(i)==0, then tfFull(i)=true,
% else =false.
% n: number of zero elements
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
%%%%

eps=2^-32;
ab=abs(a);
tfFull=ab<eps;
ma=max(ab);
if ma<eps
   tf=true;
else
    tf=false;
end
n=sum(tfFull);
end