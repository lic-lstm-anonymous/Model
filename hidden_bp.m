function [ res] = hidden_bp(res,opts)

    res.dzdx = tanh_ln(res.x,opts.dzdy);

end

