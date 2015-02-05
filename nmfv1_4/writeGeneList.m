function writeGeneList(geneList,fileName,writeDir)
% write the gene list into a .txt file
% geneList: column vector of string.
% fileName: string
% writeDir: string
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
% Feb. 25, 2013
%%%%

if nargin<3
  writeDir='.\';
end
mkdir(writeDir);
fileNameFull=[writeDir,'\',fileName,'.txt'];
fid=fopen(fileNameFull,'w');
%head=['RefGene\r\n'];
%fprintf(fid,head);

if isempty(geneList)
    fclose(fid);
    return;
end

for r=1:size(geneList,1)
  fprintf(fid,'%s\r\n',geneList{r});
end
fclose(fid);

end