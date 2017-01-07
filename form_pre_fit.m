function [ fit ] = form_pre_fit(temp2,fit)
%UNTITLED Summary of this function goes here

    seq_len=length(fit);

    for ii=1:seq_len
        
        fit{ii}.dzdx = squeeze(temp2{1}.gradAfter(:,ii,:,:));
        
    end
    
    
end

