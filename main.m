addpath('images');
imgL = imread('images/imL.png');
imgR = imread('images/imR.png');

imgL = double(imgL);
imgR = double(imgR);

% imgL = rgb2gray(imgL);
% imgR = rgb2gray(imgR);
% imgL = imresize(imgL, [64 64]);
% imgR = imresize(imgR, [64 64]);

% could be that I2 must have values ranging [0, 255] and not [0, 1]!
%imgL = im2double(imgL);
%imgR = im2double(imgR);
% imgL = double(rgb2gray(imgL))./255;
% imgR = double(rgb2gray(imgR))./255;
%imgL = double(imgL);
%imgR = double(imgR);

% Blurring image helps a bit
hgauss = fspecial('gaussian', 5, 0.6);
imgL = convn(imgL, hgauss, 'same');
imgR = convn(imgR, hgauss, 'same');

% Parameters
nDisparityValues = 16; % these images have disparity between 0 and 15.
lambda = 20.0;
nIter = 40;

% Stereo Matching with Loop Belief Propagation
tau = 15.0;
[disparity, energy] = stereoBP(imgL, imgR, nDisparityValues, lambda, tau, nIter);

figure()
plot(energy, '-o')
xlabel('Iterations'); ylabel('Energy')

figure()
imshow(disparity, [1 nDisparityValues])