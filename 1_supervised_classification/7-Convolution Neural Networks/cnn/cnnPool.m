function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.
% filter = ones(poolDim,poolDim)./(poolDim*poolDim);
% for imageNum = 1:numImages,
%     for filterNum = 1:numFilters,
%         poolImage = conv2(convolvedFeatures(:,:,filterNum,imageNum),filter,'valid');
%         poolImage = poolImage(1:poolDim:end,1:poolDim:end);
%         pooledFeatures(:,:,filterNum,imageNum) = poolImage;%./(poolDim*poolDim);
%     end
% end

numBlocks = floor(convolvedDim/poolDim);
for filterNum = 1:numFilters
    for imageNum = 1:numImages
        for poolRow = 1:numBlocks
            for poolCol = 1:numBlocks
                features = convolvedFeatures((poolRow-1)*poolDim+1:poolRow*poolDim,(poolCol-1)*poolDim+1:poolCol*poolDim,filterNum,imageNum);
                pooledFeatures(poolRow,poolCol,filterNum,imageNum)=mean(features(:));
            end
        end
    end
end

%%% YOUR CODE HERE %%%

end

