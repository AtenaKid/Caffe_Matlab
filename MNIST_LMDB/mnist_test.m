addpath('..');
use_gpu = 1;
caffe.set_mode_gpu();
gpu_id = 0;  % we will use the first gpu in this demo
caffe.set_device(gpu_id);

% Initialize the network using BVLC CaffeNet for image classification
% Weights (parameter) file needs to be downloaded from Model Zoo.
net_model = ['lenet1.prototxt'];
net_weights = ['lenet_iter_10000.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
    error('Please download CaffeNet from Model Zoo before you run this demo');
end

% Initialize a network
phase = 'test';
net = caffe.Net(net_model, net_weights, phase);

% Prepare image
for i = 1:1:1
   % im = imread(['imgTest/' num2str(i) '.jpg' ]);
    im = imread('imgTest/9.png');
    tic;
    input_data = {double(prepare_image_mnist(im))};
    toc;
    
    % do forward pass to get scores
    % scores are now Channels x Num, where Channels == 1000
    tic;
    % The net forward function. It takes in a cell array of N-D arrays
    % (where N == 4 here) containing data of input blob(s) and outputs a cell
    % array containing data from output blob(s)
    
    %input_data2 = {rand(net.blobs('data').shape)};
    
    scores = net.forward(input_data);
    toc;
    
    scores = scores{1};
    scores = mean(scores, 2);  % take average scores over 10 crops
    
    [~, maxlabel] = max(scores);
    
    display(['Number ' num2str(i) ', Network predict the image is number ' num2str(maxlabel) '!']);
end;

% call caffe.reset_all() to reset caffe
caffe.reset_all();