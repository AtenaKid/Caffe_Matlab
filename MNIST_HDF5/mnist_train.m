% demomnist
addpath('../..'); 
addpath('C:/caffe-windows');
addpath('C:/caffe-windows/matlab');

% Training
solver = caffe.Solver(['lenet_solver1.prototxt']);
train_net = solver.net;
test_net = solver.test_nets(1);

solver.solve();                                                            