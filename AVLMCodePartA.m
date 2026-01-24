% Part A number recognition
clear all; 
close all; 
clc;

%Loading data
%code to load selected folder so that i can test different image types
folderPath = uigetdir();
%loading all images in the folder
imds = imageDatastore(folderPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%shuffle to randomize testing
imds = shuffle(imds);

%Extracting features
numImages = length(imds.Files);
huMoments = zeros(numImages, 7);
labels = imds.Labels;

%loop through all images
for i = 1:numImages
    %read image
    I = readimage(imds, i);

    %convert to greyscale
    if size(I, 3) == 3; 
        I = rgb2gray(I); 
    end
    
    %adaptive thresholding
    bw = imbinarize(I, 'adaptive', 'Sensitivity', 0.5, 'ForegroundPolarity', 'dark');
    %inverting so digit is white
    bw = ~bw; 
    %removing noise
    bw = bwareaopen(bw, 30);

    %Hu moments
    moms = getHu(bw); 
    
    %transform for Knn
    huMoments(i, :) = -sign(moms) .* log10(abs(moms));
end

%Classification and splitting data
cv = cvpartition(labels, 'HoldOut', 0.3);

trainFeat = huMoments(training(cv), :);
trainLabels = labels(training(cv));
testFeat = huMoments(test(cv), :);
testLabels = labels(test(cv));

%train classifier
k = 3;
mdl = fitcknn(trainFeat, trainLabels, 'NumNeighbors', k);

%output results
predictions = predict(mdl, testFeat);
accuracy = sum(predictions == testLabels) / length(testLabels);

disp(['Accuracy (k=' num2str(k) '): ' num2str(accuracy*100) '%']);

figure;
confusionchart(testLabels, predictions);
title(['Confusion Matrix (k=' num2str(k) ')']);