function [net] = net2_init_char_lstm(net,opts)
% 
rng('default');
rng(0);

f=1/100;

n_hidden_nodes2=opts.lstm2.n_hidden_nodes;
n_input_nodes2=opts.lstm2.n_input_nodes;
n_output_nodes2=opts.lstm2.n_output_nodes;
n_cell_nodes2=opts.lstm2.n_cell_nodes;
n_gates2=opts.lstm2.n_gates;


net.Gate2.Weight=f*randn(n_gates2*n_cell_nodes2,n_hidden_nodes2+n_input_nodes2);
net.Gate2.Bias=zeros(n_gates2*n_cell_nodes2,1);
net.Input2.Weight=f*randn(n_cell_nodes2,n_hidden_nodes2+n_input_nodes2);
net.Input2.Bias=zeros(n_cell_nodes2,1);

net.Softmax2.Weight=f*randn(n_output_nodes2,n_hidden_nodes2);
net.Softmax2.Bias=zeros(n_output_nodes2,1,'single');


end




