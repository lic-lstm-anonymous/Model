function [net,cnnnet,opts, res]=train_cnn_lstm_multi(net,cnnnet,opts)

    opts.training=1;
    opts.MiniBatchError=[];
    opts.MiniBatchLoss=[];
    
    tic
    
    opts.order=randperm(opts.n_train);
    batch_size=opts.parameters.batch_size;
    
    for mini_b=1:opts.n_batch

        opts.it=opts.it+1;
        
        if opts.it == opts.momIncrease  %% first 20 minibatch uses momentum 0.5, then momentum is increased to 0.9;
            opts.parameters.mom = opts.parameters.mom2;
            opts.cnn2.mom = opts.cnn2.mom2;
        end;
        
        idx=opts.order(1+(mini_b-1)*batch_size:mini_b*batch_size);

        inputs.data=opts.train_rot(:,:,:,idx);
        inputs.labels=opts.train_labels_rot(:,idx);
        
        [net,cnnnet,res,opts] = cnn_lstm_ff_multi(net,cnnnet,inputs,opts);

        [net,cnnnet,res,opts] = cnn_lstm_bp_multi(net,cnnnet,res,opts);
        
        %forward
        
        disp([' Minibatch error: ', num2str(opts.err_origin(2)), ' Minibatch loss: ', num2str(opts.loss_origin)])
        
        opts.store.error=[opts.store.error,opts.err_origin(2)];
        opts.store.loss=[opts.store.loss,opts.loss_origin];
        opts.MiniBatchError=[opts.MiniBatchError;gather( opts.err(2))];
        opts.MiniBatchLoss=[opts.MiniBatchLoss;gather( opts.loss)];
        
        [  net.Gate2,opts ] = adam(net.Gate2,res.Gate_all2,opts);
        
        [  net.Input2,opts ] = adam(net.Input2,res.Input_all2,opts);
        
        [  net.Softmax2,opts ] = adam(net.Softmax2,res.Fit_all2,opts);
        
        [  net.Gate,opts ] = adam(net.Gate,res.Gate_all,opts);
        
        [  net.Input,opts ] = adam(net.Input,res.Input_all,opts);
        

         cnnnet.cnn2.velocity.W = opts.cnn2.mom * cnnnet.cnn2.velocity.W + opts.cnn2.alpha * res.cnngrad2{2}.W;
         cnnnet.cnn2.W = cnnnet.cnn2.W - cnnnet.cnn2.velocity.W;
         cnnnet.cnn2.velocity.b = opts.cnn2.mom * cnnnet.cnn2.velocity.b + opts.cnn2.alpha * res.cnngrad2{2}.b;
         cnnnet.cnn2.b = cnnnet.cnn2.b - cnnnet.cnn2.velocity.b;   
        
         cnnnet.velocity.W = opts.parameters.mom * cnnnet.velocity.W + opts.cnn1.alpha * res.cnngrad{2}.W;
         cnnnet.W = cnnnet.W - cnnnet.velocity.W;
         cnnnet.velocity.b = opts.parameters.mom * cnnnet.velocity.b + opts.cnn1.alpha * res.cnngrad{2}.b;
         cnnnet.b = cnnnet.b - cnnnet.velocity.b;

    end
    
    opts.results.TrainEpochError=[opts.results.TrainEpochError;mean(opts.MiniBatchError(:))];
    opts.results.TrainEpochLoss=[opts.results.TrainEpochLoss;mean(opts.MiniBatchLoss(:))];
    
    toc;

end


