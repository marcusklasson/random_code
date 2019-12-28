function [ energy ] = computeEnergy( dataCost, disparity, lambda )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

dataCostForDisparity = dataCost(disparity);
energy = sum(dataCostForDisparity(:)); % Sum of data costs

% Compute smoothness cost between neighbors
smoothnessCostU = lambda*((disparity - circshift(disparity, 1, 1)) ~= 0);
smoothnessCostD = lambda*((disparity - circshift(disparity, -1, 1)) ~= 0);
smoothnessCostL = lambda*((disparity - circshift(disparity, 1, 2)) ~= 0);
smoothnessCostR = lambda*((disparity - circshift(disparity, -1, 2)) ~= 0);

% Ignore edge costs
smoothnessCostU(1, :) = 0;
smoothnessCostD(end, :) = 0;
smoothnessCostL(:, 1) = 0;
smoothnessCostR(:, end) = 0;

% Add all smoothness costs to energy
energy = energy + sum(smoothnessCostU(:));
energy = energy + sum(smoothnessCostD(:));
energy = energy + sum(smoothnessCostL(:));
energy = energy + sum(smoothnessCostR(:));

end

