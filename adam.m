function [N,opts] = adam(N,res,opts)
%   This function is forked from matlab cnn toolbox with
%   original code link: https://github.com/xuzhenqi/cnn;
%   The function is used here for academic purpose only and complies with
%   original license;

    if ~isfield(opts.parameters,'weightDecay')
        opts.parameters.weightDecay=0;
    end
    
    
    if (~isfield(opts.parameters,'mom2'))
        opts.parameters.mom2=0.999; 
    end
    
    if ~isfield(N,'iterations')
        N.iterations=0;
    end
    
    if ~isfield(opts.results,'lrs')
        opts.results.lrs=[];%%not really necessary
    end
    opts.results.lrs=[opts.results.lrs;gather(opts.parameters.lr)];
    
    if ~isfield(opts.parameters,'eps')
        opts.parameters.eps=1e-8;
    end
    
    N.iterations=N.iterations+1;
   

   if ~isfield(N,'momentum')
      
      N.momentum{1}=zeros(size(N.Weight));
      N.momentum{2}=zeros(size(N.Bias));
      
   end

   if ~isfield(N,'momentum2')
      
      N.momentum2{1}=N.momentum{1};%initialize
      N.momentum2{2}=N.momentum{2};%initialize
                
   end

    
    mom_factor=(1-opts.parameters.mom.^N.iterations);
    mom_factor2=(1-opts.parameters.mom2.^N.iterations);
    
    
    N.momentum{1}=opts.parameters.mom.*N.momentum{1}+(1-opts.parameters.mom).*res.ac_dzdw;
    N.momentum2{1}=opts.parameters.mom.*N.momentum2{1}+(1-opts.parameters.mom).*res.ac_dzdw.^2;
    N.Weight=N.Weight-opts.parameters.lr*N.momentum{1} ...
          ./(N.momentum2{1}.^0.5+opts.parameters.eps) .*mom_factor2^0.5./mom_factor ...
          - opts.parameters.weightDecay * N.Weight;
    
    N.momentum{2}=opts.parameters.mom.*N.momentum{2}+(1-opts.parameters.mom).*res.ac_dzdb;
    N.momentum2{2}=opts.parameters.mom.*N.momentum2{2}+(1-opts.parameters.mom).*res.ac_dzdb.^2;
    N.Bias=N.Bias-opts.parameters.lr*N.momentum{2} ...
          ./(N.momentum2{2}.^0.5+opts.parameters.eps) .*mom_factor2^0.5./mom_factor;
            
        
   
end

