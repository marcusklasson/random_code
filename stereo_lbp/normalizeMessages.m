function [ msgU, msgD, msgL, msgR ] = normalizeMessages( msgU, msgD, msgL, msgR)
%UNTITLED7 Summary of this function goes here
%   Normalize messages with log-sum exp trick(?)

AU = log(sum(exp(msgU), 3));
AD = log(sum(exp(msgD), 3));
AL = log(sum(exp(msgL), 3));
AR = log(sum(exp(msgR), 3));
% 
% % Normalize by subtracting sum from messages
% msgU = bsxfun(@minus, msgU, AU);
% msgD = bsxfun(@minus, msgD, AD);
% msgL = bsxfun(@minus, msgL, AL);
% msgR = bsxfun(@minus, msgR, AR);

msgU = bsxfun(@minus, msgU, mean(msgU, 3));
msgD = bsxfun(@minus, msgD, mean(msgD, 3));
msgL = bsxfun(@minus, msgL, mean(msgL, 3));
msgR = bsxfun(@minus, msgR, mean(msgR, 3));
end

