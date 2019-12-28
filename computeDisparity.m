function [ disparity ] = computeDisparity( beliefs )
%UNTITLED2 Summary of this function goes here
%   Compute most likely disparity using argmin of beliefs
[~, disparity] = min(beliefs, [], 3);

end

