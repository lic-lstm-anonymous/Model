function cnnnet=cnnnet2_init(cnnnet,opts)
%UNTITLED Summary of this function goes here

row = opts.parameters.n_hidden_nodes + 1 - opts.cnn2.filterDim(1);
col = opts.parameters.n_frames+1- opts.cnn2.filterDim(2);
cnnnet.cnn2.paramsize = [opts.cnn2.filterDim, opts.cnn2.channel, opts.cnn2.numFilters];
cnnnet.cnn2.W=1e-1*randn(cnnnet.cnn2.paramsize);
cnnnet.cnn2.b = zeros(opts.cnn2.numFilters, 1);
cnnnet.cnn2.layersize=[row col opts.cnn2.numFilters];
cnnnet.cnn2.velocity.W=zeros(size(cnnnet.cnn2.W));
cnnnet.cnn2.velocity.b=zeros(size(cnnnet.cnn2.b));

end


