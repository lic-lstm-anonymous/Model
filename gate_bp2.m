function [res] = gate_bp2(net,res,opts)
%Summary of this function goes here

    res.dzdy = sigmoid_ln(res.y,opts.dzdy2);

    [res.dzdx, res.dzdw, res.dzdb] = backprop(res.x,net.Weight,res.dzdy);
    

end


