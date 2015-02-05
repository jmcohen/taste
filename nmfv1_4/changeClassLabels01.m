function formatedClassLabels=changeClassLabels01(originalClassLabels)
% Change class labels to 0,1,2,...
% Usage:
% formatedClassLabels=changeClassLabels01(originalClassLabels)
% originalClassLabels: column vector of numbers of string cells.
% formatedClassLabels: column vector including the 0,1,2,3,...class labels 
% for example,
% [-1;-1;1;1] to [0;0;1;1]; [-1;-1;1;1;2;2] to [0;0;1;1;2;2]
% {'normal';'normal';'cancer';'cancer'} to [0;0;1;1]
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
% May 18, 2011
%%%%


formatedClassLabels=nan(size(originalClassLabels));
uniqueLabels=unique(originalClassLabels);
for c=1:numel(uniqueLabels)
    if isnumeric(uniqueLabels)
        formatedClassLabels(originalClassLabels==uniqueLabels(c))=c-1;
    end
    if iscellstr(uniqueLabels)
        formatedClassLabels(strcmp(originalClassLabels,uniqueLabels{c}))=c-1;
    end
end

end