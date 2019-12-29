function [ dataCost ] = computeDataCost( I1, I2, nDisparityValues, tau )
%UNTITLED4 Summary of this function goes here
%   dataCost(y, x, L) is the cost for assignning label L to pixel (y, x) 

[h, w, ~] = size(I1);
dataCost = zeros([h, w, nDisparityValues]);

for i=1:nDisparityValues
    % circshift(X, l, 2) shifts X by l elements in second dims direction (x-direction)
    dataCost(:, :, i) = min(sum(abs(I1 - circshift(I2, i-1, 2)), 3)./3, tau*ones(h, w));
end

end

