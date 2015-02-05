function MAT2DAT(D,dataPointName,dataStr)
% Write a dataset in MATLAB into .dat format
% Usage:
% MAT2DAT(D,dataPointName,dataStr)
% D: matrix, each column is a multivariate data point
% dataPointName: column vector, the numeric name of each data point, for
% example, dataPointName=[0;1;1;1;0;0];
% dataStr: string, the name of the data. The output of this function is a
% .dat file that name by dataStr, for example, dataStr='myData';
% Example:
% D=randn(10,6);
% dataPointName=[0;1;1;1;0;0];
% dataStr='myData';
% MAT2DAT(D,dataPointName,dataStr);
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
% Aug. 08, 2012
%%%%


[r,c]=size(D);
fid=fopen([dataStr,'.dat'],'w');
fprintf(fid,'%d',dataPointName(1));
fprintf(fid,',%d',dataPointName(2:end));
fprintf(fid,'\t\n');
for i=1:r
    fprintf(fid,'%d',D(i,1));
    fprintf(fid,',%d',D(i,2:end));
    fprintf(fid,'\t\n');
end
fclose(fid);
end