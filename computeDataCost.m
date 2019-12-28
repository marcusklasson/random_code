function [ dataCost ] = computeDataCost( I1, I2, nDisparityValues, tau )
%UNTITLED4 Summary of this function goes here
%   dataCost(y, x, L) is the cost for assignning label L to pixel (y, x) 
tau = 15.0;
[h, w, ~] = size(I1);
dataCost = zeros([h, w, nDisparityValues]);

% could be that I2 must have values ranging [0, 255] and not [0, 1]!
for i=1:nDisparityValues
    % circshift(X, l, 2) shifts X by l elements in second dims direction (x-direction)
    %dataCost(:, :, i) = abs(sum(I1 - circshift(I2, i, 2), 3)); %abs(sum(I1 - circshift(I2, l, 2), 3))./3; 
    dataCost(:, :, i) = min(sum(abs(I1 - circshift(I2, i-1, 2)), 3)./3, tau*ones(h, w));
end

end

