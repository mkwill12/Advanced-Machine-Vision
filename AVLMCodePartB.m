% Part B Deep Learning
clear all; 
close all; 
clc;

% Loading data
folderPath = uigetdir();
imds = imageDatastore(folderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Splitting data 70/30
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

% Pre-processing
% Resize all images to 28x28 pixels and convert to RGB
inputSize = [28 28 3];
augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'ColorPreprocessing', 'gray2rgb');
augTest  = augmentedImageDatastore(inputSize, imdsTest, 'ColorPreprocessing', 'gray2rgb');

% Define Network Architecture
layers = [
    imageInputLayer(inputSize)
    
    % First Convolutional Block
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    % Second Convolutional Block
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    % Output Classification
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer
];

% Training Options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 30, ...
    'ValidationData', augTest, ...     % FIXED: Use augTest (resized)
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
% FIXED: Use augTrain (resized), not imdsTrain
net = trainNetwork(augTrain, layers, options);

% Evaluation
% FIXED: Use augTest for classification
predictedLabels = classify(net, augTest);
testLabels = imdsTest.Labels;

% Output Results
accuracy = sum(predictedLabels == testLabels) / length(testLabels);
disp(['CNN Accuracy: ' num2str(accuracy*100) '%']);

figure;
confusionchart(testLabels, predictedLabels);
title('CNN Confusion Matrix');