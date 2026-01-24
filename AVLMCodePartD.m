% CW1 Part D: Final Extraction (Standard Digits)
close all; 
clc;

if ~exist('net', 'var'); error('Run Part B first!'); end

% 1. Load Image
disp('Select your BLOCK DIGIT image (e.g. 7 2 4)...');
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg'});
if isequal(file, 0); return; end
I = imread(fullfile(path, file));

% 2. Pre-processing
if size(I, 3) == 3; Igray = rgb2gray(I); else; Igray = I; end

% Standard Detection
bw = imbinarize(Igray, 'adaptive', 'Sensitivity', 0.5, 'ForegroundPolarity', 'dark');
bw = ~bw; 
bw = imclose(bw, strel('disk', 4)); % Connect broken lines
bw = bwareaopen(bw, 200); % Remove noise

% 3. Get Blobs
blobs = regionprops(bw, 'BoundingBox', 'Area', 'Image');

% 4. Filter
validBlobs = {}; count = 0;
for k = 1:length(blobs)
    % Simple size filter for clean paper
    if blobs(k).Area > 500
        count = count + 1;
        validBlobs{count} = blobs(k);
    end
end

if count == 0; msgbox('No digits found!'); return; end

% 5. Sort Left-to-Right
xCoords = zeros(1, count);
for i = 1:count; xCoords(i) = validBlobs{i}.BoundingBox(1); end
[~, sortIndex] = sort(xCoords);

% 6. Classify
detectedString = "";
figure; imshow(I); title('Part D: Final Sequence Extraction'); hold on;

for i = 1:count
    idx = sortIndex(i);
    thisBox = validBlobs{idx}.BoundingBox;
    
    % Square Crop (Prevents Squashing)
    maxSide = max(thisBox(3), thisBox(4));
    squareSize = maxSide * 1.4; 
    
    centerX = thisBox(1) + thisBox(3)/2;
    centerY = thisBox(2) + thisBox(4)/2;
    
    x = floor(centerX - squareSize/2);
    y = floor(centerY - squareSize/2);
    s = floor(squareSize);
    
    x = max(x, 1); y = max(y, 1);
    w = min(s, size(I, 2) - x); h = min(s, size(I, 1) - y);
    
    % Crop
    subImage = Igray(y:y+h, x:x+w);
    
    % --- PROCESS FOR AI ---
    % 1. Invert (Make ink white, paper black)
    subImage = imcomplement(subImage);
    
    % 2. Contrast Stretch (Make it bold)
    subImage = imadjust(subImage);
    
    % 3. Resize to 28x28
    imgResized = imresize(subImage, [28 28]);
    
    % 4. Make RGB
    imgFinal = cat(3, imgResized, imgResized, imgResized);
    
    % Classify
    predictedLabel = char(classify(net, imgFinal));
    detectedString = detectedString + predictedLabel;
    
    % Draw Result
    rectangle('Position', thisBox, 'EdgeColor', 'c', 'LineWidth', 3);
    text(thisBox(1), thisBox(2)-30, predictedLabel, ...
        'Color', 'r', 'FontSize', 30, 'FontWeight', 'bold');
end
hold off;

% Output
msgbox(['Detected Sequence: ' char(detectedString)], 'Success');
disp(['Final Result: ' char(detectedString)]);