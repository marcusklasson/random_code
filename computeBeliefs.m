function [ beliefs ] = computeBeliefs(dataCost, msgU, msgD, msgL, msgR)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

beliefs = dataCost;
% Incoming messages
incomingMsgFromU = circshift(msgD, 1, 1);
incomingMsgFromD = circshift(msgU, -1, 1);
incomingMsgFromL = circshift(msgR, 1, 2);
incomingMsgFromR = circshift(msgL, -1, 2);

beliefs = beliefs + incomingMsgFromU + incomingMsgFromD + incomingMsgFromL + incomingMsgFromR;

end

