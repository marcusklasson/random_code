addpath('images');
imgL = imread('images/imL.png');
imgR = imread('images/imR.png');

imgL = double(imgL);
imgR = double(imgR);

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