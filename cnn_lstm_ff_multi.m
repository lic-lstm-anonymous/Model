function [ net,cnnnet,res,opts ] = cnn_lstm_ff_multi( net,cnnnet,preinputs,opts )
%Summary of this function goes here

res.temp = cell(2, 1);
res.grad = cell(2, 1);
res.temp{1}.after = preinputs.data;
[res.temp{2}.after, res.temp{2}.linTrans] = cnnConvolve(res.temp{1}.after, cnnnet.W, cnnnet.b, opts.cnn1.nonLinearType);
res.temp{2}.sizeafter=size(res.temp{2}.after);
res.temp{2}.after_for_backprop=res.temp{2}.after;
res.temp{2}.after=permute(squeeze(res.temp{2}.after),[2,3,1]);
inputs.data=res.temp{2}.after;
inputs.labels=preinputs.labels(opts.win_len:end,:)';


    n_frames=opts.parameters.n_frames;
    
    n_cell_nodes=opts.parameters.n_cell_nodes;
    n_hidden_nodes=opts.parameters.n_hidden_nodes;
    batch_size=opts.parameters.batch_size;
    
    res.Cell{1}.x=zeros(n_cell_nodes,batch_size,'like',inputs.data);
    res.Hidden{1}.x=zeros(n_hidden_nodes,batch_size,'like',inputs.data);


    for f=1:n_frames

        res.Gate{f}.x=[res.Hidden{f}.x;inputs.data(:,:,f)];    %%%% first dimension of input data is 67, second dimension is batch size, third dimension is time frame;
        res.Input{f}.x=res.Gate{f}.x;
        
        [res.Gate{f}.z,res.Gate{f}.y] = gate_ff(net.Gate,res.Gate{f});
        
        [res.Input{f}.z,res.Input{f}.y] = input_ff(net.Input,res.Input{f});
        
        res.Cell{f+1}.x=res.Gate{f}.z(1:n_cell_nodes,:).*res.Input{f}.z+res.Gate{f}.z(n_cell_nodes+1:2*n_cell_nodes,:).*res.Cell{f}.x;
        
        [res.Cell{f+1}.z] = tanh_ff(res.Cell{f+1});
        
        res.Hidden{f+1}.x=res.Gate{f}.z(2*n_cell_nodes+1:3*n_cell_nodes,:).*res.Cell{f+1}.z;
    
        res.Fit{f}.x=res.Hidden{f+1}.x;
        
    end
    

    secoinputs.data=formnewdata(res.Hidden,opts.cnn2.channel);
    res.temp2 = cell(2, 1);
    res.grad2 = cell(2, 1);
    res.temp2{1}.after = secoinputs.data;
    [res.temp2{2}.after, res.temp2{2}.linTrans] = cnnConvolve(res.temp2{1}.after, cnnnet.cnn2.W, cnnnet.cnn2.b, opts.cnn2.nonLinearType);
    res.temp2{2}.sizeafter=size(res.temp2{2}.after);
    res.temp2{2}.after_for_backprop=res.temp2{2}.after;
    res.temp2{2}.after=permute(squeeze(res.temp2{2}.after),[2,3,1]);
    inputs2.data=res.temp2{2}.after;
    inputs2.labels=preinputs.labels((opts.win_len+opts.cnn2.win_len-1):end,:)';

    assert(size(inputs2.data,3)==size(inputs2.labels,2),'the length of input data and the input label is not the same;');
    n_frames_2=opts.lstm2.n_frames;
    
    n_cell_nodes2=opts.lstm2.n_cell_nodes;
    n_hidden_nodes2=opts.lstm2.n_hidden_nodes;
    
    res.Cell2{1}.x=zeros(n_cell_nodes2,batch_size,'like',inputs2.data);
    res.Hidden2{1}.x=zeros(n_hidden_nodes2,batch_size,'like',inputs2.data);
    
    opts.err=zeros(1,n_frames_2,'like',inputs2.data);
    if isfield(inputs2,'labels')
        opts.err=zeros(2,n_frames_2,'like',inputs2.data);
        opts.loss=zeros(1,n_frames_2,'like',inputs2.data);
    end
    
    f=-2;
    opts.cost=0;
    extLabels=cell(n_frames_2,1);
    for g=1:n_frames_2    
        
        extLabels{g} = zeros(opts.emb_len, batch_size);

        extLabels{g}(sub2ind(size(extLabels{g}), inputs2.labels(:,g)', 1 : batch_size)) = 1;
    
        res.Gate2{g}.x=[res.Hidden2{g}.x;inputs2.data(:,:,g)]; 
    
        res.Input2{g}.x=res.Gate2{g}.x;
        
        [res.Gate2{g}.z,res.Gate2{g}.y] = gate_ff(net.Gate2,res.Gate2{g});
        
        [res.Input2{g}.z,res.Input2{g}.y] = input_ff(net.Input2,res.Input2{g});
        
        res.Cell2{g+1}.x=res.Gate2{g}.z(1:n_cell_nodes2,:).*res.Input2{g}.z+res.Gate2{g}.z(n_cell_nodes2+1:2*n_cell_nodes2,:).*res.Cell2{g}.x;
        
        [res.Cell2{g+1}.z] = tanh_ff(res.Cell2{g+1});
    
        res.Hidden2{g+1}.x=res.Gate2{g}.z(2*n_cell_nodes2+1:3*n_cell_nodes2,:).*res.Cell2{g+1}.z;
        
        res.Fit2{g}.x=res.Hidden2{g+1}.x;
        
        res.Fit2{g}.class=inputs2.labels(:,g);
        
        [res.Fit2{g}.z,res.Fit2{g}.y] = soft_ff(net.Softmax2,res.Fit2{g});
        
        opts.cost=opts.cost- mean(sum(extLabels{g} .* log(res.Fit2{g}.z)));
        
        if isfield(inputs2,'labels')
            opts.err(:,g)=error_multiclass(res.Fit2{g}.class,res.Fit2{g}.z);

        end
        
    end
    
    opts.err_origin=mean(opts.err,2)./batch_size;
    opts.loss_origin=opts.cost;
    
end

