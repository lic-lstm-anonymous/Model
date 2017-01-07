function [convolvedFeatures, linTrans] = cnnConvolve(images, W, b, nonlineartype, con_matrix, shape)
%   This function is forked from matlab cnn toolbox with
%   original code link: https://github.com/xuzhenqi/cnn;
%   The function is used here for academic purpose only and complies with
%   original license;

[filterDimRow,filterDimCol,channel,numFilters] = size(W);

if ~exist('con_matrix','var') || isempty(con_matrix)
    con_matrix = ones(channel, numFilters);
end

if ~exist('nonlineartype','var')
    nonlineartype = 'sigmoid';
end

if ~exist('shape','var')
    shape = 'valid';
end

[imageDimRow, imageDimCol,~, numImages] = size(images);
convDimRow = imageDimRow - filterDimRow + 1;
convDimCol = imageDimCol - filterDimCol + 1;

convolvedFeatures = zeros(convDimRow, convDimCol, numFilters, numImages); % the input features to the next layer;

for imageNum = 1:numImages
  for filterNum = 1:numFilters
      convolvedImage = zeros(convDimRow, convDimCol);
      for channelNum = 1:channel
          if con_matrix(channelNum,filterNum) ~= 0
            % Obtain the feature (filterDim x filterDim) needed during the convolution
            filter = W(:,:,channelNum,filterNum);

            % Flip the feature matrix because of the definition of convolution, as explained later
            filter = rot90(squeeze(filter),2); % the rotating has the effect that the weight is multiplied directly with each region;

            % Obtain the image
            im = squeeze(images(:, :, channelNum,imageNum));

            convolvedImage = convolvedImage + conv2(im, filter, shape); % in the first convolution layer, the three channels are added together; 
          end
            % Add the bias unit     
      end
      convolvedImage = convolvedImage + b(filterNum); % the dimension of bias equals the number of filters; 
      convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end
linTrans = convolvedFeatures;
switch nonlineartype
    case 'sigmoid'
        convolvedFeatures = 1./(1+exp(-convolvedFeatures));
    case 'relu'
        convolvedFeatures = max(0,convolvedFeatures);
    case 'tanh'
        convolvedFeatures = tanh(convolvedFeatures);
    case 'softsign'
        convolvedFeatures = convolvedFeatures ./ (1 + abs(convolvedFeatures));
    case 'none'
    otherwise
        fprintf('error: no such nonlieartype%s',nonlineartype);
end
end

