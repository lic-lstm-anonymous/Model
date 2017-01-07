function res = soft_bp2(net,res,opts)
%  Summary of this function goes here

for f=1:opts.lstm2.n_frames

    res.Fit2{f}.dzdy = softloglossbp(res.Fit2{f}.y,res.Fit2{f}.class);  %% this propagates to output of hidden unit;

    [res.Fit2{f}.dzdx, res.Fit2{f}.dzdw,res.Fit2{f}.dzdb] = backprop(res.Fit2{f}.x,net.Softmax2.Weight,res.Fit2{f}.dzdy);
    
end
    
end


