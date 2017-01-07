function [ res] = hidden_bp2(res,opts)

    res.dzdx = tanh_ln(res.x,opts.dzdy2);

end

