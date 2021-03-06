function [ disparity, energy ] = stereoBP( I1, I2, nDisparityValues, lambda, tau, nIter )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

dataCost = computeDataCost(I1, I2, nDisparityValues, tau);
energy = zeros(nIter, 1);

[h, w, ~] = size(I1);
% Messages sent to neighbors in each direction (up, down, left, right)
msgU = zeros([h, w, nDisparityValues]);
msgD = zeros([h, w, nDisparityValues]);
msgL = zeros([h, w, nDisparityValues]);
msgR = zeros([h, w, nDisparityValues]);

for i = 1:nIter
    
    [msgU, msgD, msgL, msgR] = updateMessages(msgU, msgD, msgL, msgR, dataCost, lambda, 'potts_model');
    % Normalize messages
    [msgU, msgD, msgL, msgR] = normalizeMessages(msgU, msgD, msgL, msgR);
    
    % Compute belief's
    beliefs = computeBeliefs(dataCost, msgU, msgD, msgL, msgR);
    
    % Compute labeling of disparity
    disparity = computeDisparity(beliefs);
    
    % Compute energy cost
    energy(i) = computeEnergy(dataCost, disparity, lambda);
    fprintf('Iteration %i Energy: %3.4f \n', i, energy(i))
    
end
fprintf('Algorithm done. \n\n')

end

