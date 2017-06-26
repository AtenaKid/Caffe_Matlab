function images = prepare_image_mnist(im)

IMAGE_DIM = 28;

% If input image is too big , is rgb and of type uint8:
% -> resize to fixed input size, single channel, type float
if size(im, 3) == 3
im = rgb2gray(im);
end;
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
im = single(im);

% Caffe needs a 4D input matrix which has single precision
% Data has to be scaled by 1/256 = 0.00390625 (like during training)
% In the second last line the image is beeing transposed!
images = zeros(IMAGE_DIM,IMAGE_DIM, 1, 1);
images(:,:, 1, 1) = 0.00390625*im';
images = single(images);