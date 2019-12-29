function [ msgU, msgD, msgL, msgR ] = updateMessages( msgUPrev, msgDPrev, msgLPrev, msgRPrev, dataCost, lambda, smoothness )
%UNTITLED5 Summary of this function goes here
%   Min-Sum message update

% Where does SmoothnessCost comes in?! Probably has something to with the
% Potts model!
msgU = zeros(size(dataCost));
msgD = zeros(size(dataCost));
msgL = zeros(size(dataCost));
msgR = zeros(size(dataCost));

[h, w, nDisparityValues] = size(dataCost);

% Remove incoming messages on corresponding borders by settign them to zero?
incomingMsgFromU = circshift(msgDPrev, 1, 1);
incomingMsgFromD = circshift(msgUPrev, -1, 1);
incomingMsgFromL = circshift(msgRPrev, 1, 2);
incomingMsgFromR = circshift(msgLPrev, -1, 2);

% Sum DataCost and incoming messages
npqU = dataCost + incomingMsgFromD + incomingMsgFromL + incomingMsgFromR;
npqD = dataCost + incomingMsgFromU + incomingMsgFromL + incomingMsgFromR;
npqL = dataCost + incomingMsgFromD + incomingMsgFromU + incomingMsgFromR;
npqR = dataCost + incomingMsgFromD + incomingMsgFromL + incomingMsgFromU;

if strcmp(smoothness, 'truncated_linear')
    K = 2; %truncation value
    
    for i=1:nDisparityValues
        msgUTmp = zeros(size(dataCost));
        msgDTmp = zeros(size(dataCost));
        msgLTmp = zeros(size(dataCost));
        msgRTmp = zeros(size(dataCost));

        for j=1:nDisparityValues
            msgUTmp(:, :, j) = npqU(:, :, j) + lambda*min(abs(i-j), K);
            msgDTmp(:, :, j) = npqD(:, :, j) + lambda*min(abs(i-j), K);
            msgLTmp(:, :, j) = npqL(:, :, j) + lambda*min(abs(i-j), K);
            msgRTmp(:, :, j) = npqR(:, :, j) + lambda*min(abs(i-j), K);
        end

        msgU(:, :, i) = min(msgUTmp, [], 3);
        msgD(:, :, i) = min(msgDTmp, [], 3);
        msgL(:, :, i) = min(msgLTmp, [], 3);
        msgR(:, :, i) = min(msgRTmp, [], 3);
    end

elseif  strcmp(smoothness, 'potts_model') % Potts model
    
    % Get smallest disparity values
    spqU = min(npqU, [], 3);
    spqD = min(npqD, [], 3);
    spqL = min(npqL, [], 3);
    spqR = min(npqR, [], 3);

    % Potts model
    for i=1:nDisparityValues
        msgU(:, :, i) = min(npqU(:, :, i), lambda + spqU);
        msgD(:, :, i) = min(npqD(:, :, i), lambda + spqD);
        msgL(:, :, i) = min(npqL(:, :, i), lambda + spqL);
        msgR(:, :, i) = min(npqR(:, :, i), lambda + spqR);

    end

end

end

