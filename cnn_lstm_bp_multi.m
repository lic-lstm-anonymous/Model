function [net,cnnnet,res,opts] = cnn_lstm_bp_multi(net,cnnnet,res,opts)
%Summary of this function goes here

n_frames_2=opts.lstm2.n_frames;
n_cell_nodes2=opts.lstm2.n_cell_nodes;
n_hidden_nodes2=opts.lstm2.n_hidden_nodes;
batch_size=opts.parameters.batch_size;

res=soft_bp2(net,res,opts);

dzdct2=0;
res.Gate2{n_frames_2+1}.dzdx=zeros(size(res.Gate2{n_frames_2}.x));
res.Input2{n_frames_2+1}.dzdx=zeros(size(res.Gate2{n_frames_2}.x));


for f=n_frames_2:-1:1

opts.dzdy2=res.Gate2{f}.z(2*n_cell_nodes2+1:3*n_cell_nodes2,:).*(res.Fit2{f}.dzdx+res.Gate2{f+1}.dzdx(1:size(res.Hidden2{f}.x,1),:)+res.Input2{f+1}.dzdx(1:size(res.Hidden2{f}.x,1),:));
res.Cell2{f+1} = hidden_bp2(res.Cell2{f+1},opts);
res.Cell2{f+1}.dzdx=res.Cell2{f+1}.dzdx+dzdct2;
dzdct2=res.Cell2{f+1}.dzdx.*res.Gate2{f}.z(n_cell_nodes2+1:2*n_cell_nodes2,:);

opts.dzdy2= res.Gate2{f}.z(1:n_cell_nodes2,:).*res.Cell2{f+1}.dzdx;
[res.Input2{f}] = input_bp2(net.Input2,res.Input2{f},opts);

opts.dzdy2=[res.Input2{f}.z.*res.Cell2{f+1}.dzdx;...
res.Cell2{f}.x .*res.Cell2{f+1}.dzdx;...
res.Fit2{f}.dzdx.*res.Cell2{f+1}.z];
res.Gate2{f} = gate_bp2(net.Gate2,res.Gate2{f},opts);


end

res.Cell2{1}.dzdx=0;

    [res.Fit_all2.ac_dzdw,res.Fit_all2.ac_dzdb]=average_gradients2(res.Fit2,opts);
    [res.Input_all2.ac_dzdw,res.Input_all2.ac_dzdb]=average_gradients2(res.Input2,opts);
    [res.Gate_all2.ac_dzdw,res.Gate_all2.ac_dzdb]=average_gradients2(res.Gate2,opts);



     res.temp2{2}.gradtemp=zeros(res.temp2{2}.sizeafter);
     seq_len_2=opts.lstm2.n_frames;
     for ii=1:seq_len_2
     
         res.Input2{ii}.dzdx_x=res.Input2{ii}.dzdx((opts.lstm2.n_hidden_nodes+1):end,:)+res.Gate2{ii}.dzdx((opts.lstm2.n_hidden_nodes+1):end,:);
     
         res.temp2{2}.gradtemp(1,ii,:,:)=res.Input2{ii}.dzdx_x;
         
     end

     switch opts.cnn2.nonLinearType
     
         case 'sigmoid'
         
             res.temp2{2}.gradBefore=res.temp2{2}.gradtemp.*res.temp2{2}.after_for_backprop.*(1 - res.temp2{2}.after_for_backprop);
     
     end
     
     tempW2 = zeros([size(cnnnet.cnn2.W) batch_size]);
     numInputMap2 = size(tempW2, 3);
     numOutputMap2 = size(tempW2, 4);
     for i = 1 : batch_size
         for nI = 1 : numInputMap2
             for nO = 1 : numOutputMap2
                 tempW2(:,:,nI,nO,i) = conv2(res.temp2{1}.after(:,:,nI,i), rot90(res.temp2{2}.gradBefore(:,:,nO,i), 2), 'valid');   
             end
         end
     end
     
     res.cnngrad2{2}.W = mean(tempW2,5); % 
     tempb2 = mean(sum(sum(res.temp2{2}.gradBefore)),4);
     res.cnngrad2{2}.b = tempb2(:);
     
     
     numChannel2 = size(res.temp2{1}.after,3);
     res.temp2{1}.gradAfter = zeros(size(res.temp2{1}.after));
     for i = 1 : batch_size
         for c = 1 : numChannel2
             for j = 1 : size(res.temp2{2}.gradBefore, 3)
             
                 res.temp2{1}.gradAfter(:,:,c,i) = res.temp2{1}.gradAfter(:,:,c,i) + conv2(res.temp2{2}.gradBefore(:,:,j,i), cnnnet.cnn2.W(:,:,c,j), 'full');
                 
             end
         end
     end
     
     res.Fit=form_pre_fit(res.temp2,res.Fit);
     
     
    n_frames=opts.parameters.n_frames;
    n_cell_nodes=opts.parameters.n_cell_nodes;
    n_hidden_nodes=opts.parameters.n_hidden_nodes;
    batch_size=opts.parameters.batch_size;

    %2: BPTT: calculate the gradient wrt memory cell 

    dzdct=0;%accumulated gradient in later time frames
    res.Gate{n_frames+1}.dzdx=zeros(size(res.Gate{n_frames}.x));
    res.Input{n_frames+1}.dzdx=zeros(size(res.Input{n_frames}.x));
    for f=n_frames:-1:1
        %%the gradient here is the previous accumulated gradient + the one from the output gate 
        
        opts.dzdy=res.Gate{f}.z(2*n_cell_nodes+1:3*n_cell_nodes,:).*(res.Fit{f}.dzdx+res.Gate{f+1}.dzdx(1:size(res.Hidden{f}.x,1),:)+res.Input{f+1}.dzdx(1:size(res.Hidden{f}.x,1),:));
        res.Cell{f+1} = hidden_bp(res.Cell{f+1},opts);
        res.Cell{f+1}.dzdx=res.Cell{f+1}.dzdx+dzdct;
        %%bp to previous time frame
        dzdct=res.Cell{f+1}.dzdx.*res.Gate{f}.z(n_cell_nodes+1:2*n_cell_nodes,:);
        
        opts.dzdy= res.Gate{f}.z(1:n_cell_nodes,:).*res.Cell{f+1}.dzdx;
        [res.Input{f}] = input_bp(net.Input,res.Input{f},opts);
        
        
        opts.dzdy=[res.Input{f}.z.*res.Cell{f+1}.dzdx;...
        res.Cell{f}.x .*res.Cell{f+1}.dzdx;...
        res.Fit{f}.dzdx.*res.Cell{f+1}.z];
        res.Gate{f} = gate_bp(net.Gate,res.Gate{f},opts);
        
    end
    
    %res.Cell{1}(end+1).x=0;  %just some padding
    res.Cell{1}.dzdx=0;
    


    %%%accumulate gradients in all time frames
    
    [res.Input_all.ac_dzdw,res.Input_all.ac_dzdb]=average_gradients(res.Input,opts);
    [res.Gate_all.ac_dzdw,res.Gate_all.ac_dzdb]=average_gradients(res.Gate,opts);


    res.temp{2}.gradtemp=zeros(res.temp{2}.sizeafter);
    
    seq_len=opts.parameters.n_frames;
    for ii=1:seq_len
        res.Input{ii}.dzdx_x=res.Input{ii}.dzdx((opts.parameters.n_hidden_nodes+1):end,:)+res.Gate{ii}.dzdx((opts.parameters.n_hidden_nodes+1):end,:);
        res.temp{2}.gradtemp(1,ii,:,:)=res.Input{ii}.dzdx_x;
    
    end
    
    switch opts.cnn1.nonLinearType
        case 'sigmoid'

            res.temp{2}.gradBefore=res.temp{2}.gradtemp.*res.temp{2}.after_for_backprop.*(1 - res.temp{2}.after_for_backprop);
        
    end
    
    tempW = zeros([size(cnnnet.W) batch_size]);
    numInputMap = size(tempW, 3);
    numOutputMap = size(tempW, 4);
    for i = 1 : batch_size
        for nI = 1 : numInputMap
            for nO = 1 : numOutputMap
                
                tempW(:,:,nI,nO,i) = conv2(res.temp{1}.after(:,:,nI,i), rot90(res.temp{2}.gradBefore(:,:,nO,i), 2), 'valid');   

            end
        end
    end
    
    res.cnngrad{2}.W = mean(tempW,5);
    tempb = mean(sum(sum(res.temp{2}.gradBefore)),4);
    res.cnngrad{2}.b = tempb(:);

end


