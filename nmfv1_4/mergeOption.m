function optionFinal=mergeOption(option,optionDefault)
% Merge two struct options into one struct
% Usage:
% optionFinal=mergeOption(option,optionDefault)
% option: struct
% optionDefault: struct
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
% Mar. 18, 2011
%%%%

optionFinal=optionDefault;
if isempty(option)
    return;
end
names=fieldnames(option);
for i=1:numel(names)
    optionFinal=setfield(optionFinal,names{i},getfield(option,names{i}));
end
end