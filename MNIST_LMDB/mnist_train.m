% demomnist
addpath('../..'); 
addpath('C:/caffe-windows');
addpath('C:/caffe-windows/matlab');

% Training
solver = caffe.Solver(['lenet_solver1.prototxt']);
solver.solve();                                                            