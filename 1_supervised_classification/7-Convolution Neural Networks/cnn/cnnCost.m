function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

lambda = 1;

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
%1.convolved
activations = cnnConvolve(filterDim,numFilters,images,Wc,bc);
% for imageNum = 1:numImages,
%     for filtersNum = 1:numFilters,
%         convolvedImage = zeros(convDim,convDim);
%         filter = Wc(:,:,filtersNum);
%         filter = rot90(squeeze(filter),2);
%         im = squeeze(images(:,:,imageNum));
%         convolvedImage = conv2(im,filter,'valid');
%         convolvedImage = convolvedImage + b(filterNum);
%         convolvedImage = 1./(1+exp(-convolvedImage));
%         activations(:,:,filtersNum,imageNum) = convolvedImage;
%     end
% end
%2.pooled
activationsPooled = cnnPool(poolDim,activations);
% for imageNum = 1:numImages,
%     for filtersNum = 1:numFilters,
%         for poolRow = 1:numBlocks,
%             for poolCol = 1:numBlocks
%                 features = activations((poolRow-1)*poolDim +1:poolRow*poolDim,(poolCol-1)*poolDim+1:poolCol*poolDim,filterNum,imageNum);
%                 activationsPooled(poolRow,poolCol,filterNum,imageNum) = mean(features(:));
%             end
%         end
%     end
% end


% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);

%%% YOUR CODE HERE %%%
%3.softmax layer
h = exp(bsxfun(@plus,Wd*activationsPooled,bd));
probs = bsxfun(@rdivide,h,sum(h,1));
%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.

cost = 0; % save objective into cost

%%% YOUR CODE HERE %%%
logp = log(probs);
index = sub2ind(size(logp),labels',1:size(probs,2));
ceCost = -sum(logp(index));
wCost = lambda/2 * (sum(Wd(:).^2)+sum(Wc(:).^2));
cost = ceCost/numImages + wCost;


% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
%1.softmax layer
output = zeros(size(probs));
output(index) = 1;
DeltaSoftmax = probs - output;
%2.pool layer
DeltaPool = reshape(Wd'*DeltaSoftmax,outputDim,outputDim,numFilters,numImages);
DeltaUnpool = zeros(convDim,convDim,numFilters,numImages);
for imNum = 1:numImages
    for FilterNum = 1:numFilters
        unpool = DeltaPool(:,:,FilterNum,imNum);
        DeltaUnpool(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim,poolDim))./(poolDim^2);
    end
end
%3.convolution layer
DeltaConv = DeltaUnpool.* activations.*(1-activations);
%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
Wd_grad = (1./numImages).*DeltaSoftmax*activationsPooled'+lambda*Wd;
bd_grad = (1./numImages).*sum(DeltaSoftmax,2);

bc_grad = zeros(size(bc));
Wc_grad = zeros(filterDim,filterDim,numFilters);

for filterNum = 1:numFilters
    error = DeltaConv(:,:,filterNum,:);
    bc_grad(filterNum) = (1./numImages).*sum(error(:));
end

for filterNum = 1:numFilters
    for imNum = 1:numImages
        error = DeltaConv(:,:,filterNum,imNum);
        DeltaConv(:,:,filterNum,imNum) = rot90(error,2);
    end
end

for filterNum = 1:numFilters
    for imNum = 1:numImages
        Wc_grad(:,:,filterNum) = Wc_grad(:,:,filterNum)+conv2(images(:,:,imNum),DeltaConv(:,:,filterNum,imNum),'valid');
    end
end

Wc_grad = (1./numImages).*Wc_grad + lambda*Wc;

%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
