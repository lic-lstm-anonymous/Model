function [res] = input_bp2( net,res,opts )
%   Summary of this function goes here
%   Detailed explanation goes here

    res.dzdy = tanh_ln(res.y,opts.dzdy2);
    
    [res.dzdx, res.dzdw,res.dzdb] = backprop(res.x,net.Weight,res.dzdy);

end

