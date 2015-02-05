function [sortedDist,index] = getBestScores(dist,k)
% get the nearest K values from a distance vector
% dist; column vector
% k: scalar
% sortedDist: column vector, the k best distances
% index: the numeric index of these k best distances
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
% May 20, 2011
%%%%

% sort if needed
if k>1
    [sortedDist,index] = sort(dist,'descend'); % sorted from the greatest to the lowest
    sortedDist = sortedDist(1:k); % Get the nearest k elements
    index = index(1:k);   % Get the corresponding index in dist
else
    [sortedDist,index] = max(dist);
end