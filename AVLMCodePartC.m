% CW1 Part C: Simultaneous Detection and Classification
close all; 
clc;

% 1. Load Data
disp('Select the detection image...');
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg'});
if isequal(file, 0); return; end
I = imread(fullfile(path, file));

% 2. Pre-processing
if size(I, 3) == 3
    Igray = rgb2gray(I);
else
    Igray = I;
end

% Adaptive thresholding to handle uneven lighting
bw = imbinarize(Igray, 'adaptive', 'Sensitivity', 0.4, 'ForegroundPolarity', 'dark');
bw = ~bw; % Invert so digit is white

% Morphological dilation to connect broken strokes
% This helps merge the top and bottom of digits like '5'
se = strel('disk', 5); 
bw = imdilate(bw, se);

% Remove small noise
bw = bwareaopen(bw, 1000);

% 3. Detection
% Extract blob properties for filtering
blobs = regionprops(bw, 'BoundingBox', 'Area', 'Solidity', 'Image');

figure; 
imshow(I); 
title('Part C Result'); 
hold on;

% Loop through all detected blobs
for k = 1:length(blobs)
    thisBox = blobs(k).BoundingBox;
    thisArea = blobs(k).Area;
    thisSolidity = blobs(k).Solidity;
    
    % Calculate Aspect Ratio (Height / Width)
    w = thisBox(3);
    h = thisBox(4);
    ratio = h / w; 
    
    % Filters (Geometric properties)
    % 1. Aspect Ratio: Reject long thin lines (e.g., door gaps)
    isNotLine = ratio < 4; 
    
    % 2. Solidity: Reject solid blocks (e.g., door handles)
    isNotSolid = thisSolidity < 0.75;
    
    % 3. Area: Reject tiny noise or massive objects
    isRightSize = thisArea > 1000 && thisArea < 10000;

    if isRightSize && isNotSolid && isNotLine
        
        % Crop with padding to preserve edges
        padding = 10;
        x = max(floor(thisBox(1)) - padding, 1);
        y = max(floor(thisBox(2)) - padding, 1);
        wBox = min(thisBox(3) + 2*padding, size(I, 2) - x);
        hBox = min(thisBox(4) + 2*padding, size(I, 1) - y);
        
        subImage = I(y:y+hBox, x:x+wBox, :);
        
        % Resize to 28x28 for the CNN
        imgResized = imresize(subImage, [28 28]);
        
        % Ensure RGB format for the network
        if size(imgResized, 3) == 1
            imgResized = cat(3, imgResized, imgResized, imgResized);
        end
        
        % Classification
        predictedLabel = classify(net, imgResized);
        
        % Visualization
        rectangle('Position', thisBox, 'EdgeColor', 'g', 'LineWidth', 3);
        text(thisBox(1), thisBox(2)-15, char(predictedLabel), ...
            'Color', 'y', 'FontSize', 18, 'FontWeight', 'bold', 'BackgroundColor', 'k');
    end
end
hold off;