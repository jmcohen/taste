function [gm]=geometricMean(v)
% geometricMean of vector v
% for example
% v=[1,2,3,NaN];
% [gm]=geometricMean(v)
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
% July 04, 2012
%%%%

len=numel(v);
lenNaN=len;
p=1; % product
for i=1:len
   if isnan(v(i))
      lenNaN=lenNaN-1;
      continue;
   end
   p=p*v(i);
end
if lenNaN>0
   gm=nthroot(p,lenNaN);
else
    gm=NaN;
end
end