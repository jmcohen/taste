function Hinv=pseudoinverse(H,lambda)
% compute pseudoinverse of matrix H
% Usage:
% Hinv=pseudoinverse(H);
% Hinv=pseudoinverse(H,lambda);
% H: matrix
% lambda: scalar, the parameter to reach stable result and avoid
% singularity, its value should be very large, i.e. 2^32.
% Hinv, the pseudoinverse of H
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
% May 01, 2011
%%%%

[r,c]=size(H);
if nargin<2
    if r<c
        Hinv=(H'*H)\H';
    else
        Hinv=H'/(H'*H);
    end
    
    return;
end
if r<c
    Hinv=((1/lambda).*eye(c,c)+H'*H)\H';
else
    Hinv=H'/((1/lambda).*eye(r,r)+H*H');
end
end