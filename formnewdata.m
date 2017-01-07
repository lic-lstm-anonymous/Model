function data = formnewdata(Hidden,channel)
%UNTITLED Summary of this function goes here

len=length(Hidden)-1;
[lstmsize,batchsize]=size(Hidden{1}.x);

data=zeros(lstmsize,len,channel,batchsize);

for i=1:len
    for j=1:channel
        data(:,i,j,:)=Hidden{i+1}.x;
    end
end



end

