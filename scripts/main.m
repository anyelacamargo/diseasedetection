% This program segments visual symptoms of plant diseases
% Anyela Camargo, August 2016.

function main(imagefolder)
    %change these locations as needed
    rootname =  'diseasedetection';
    resultf =  pwd();
    %imagefolder = 'images';
    outputfileLession = 'lession.csv';
    DiseaseFeatures(rootname, imagefolder, outputfileLession, resultf);


 % Extract disease features
 function DiseaseFeatures(rootname, imagefolder, outputfileLession, resultf)
    %rd = dir(strcat('..\', imagefolder, '\','*.jpg'));    
    rd = dir(fullfile(imagefolder, '*.jpg'));
    %fileID = fopen(char(strcat(resultf,'\', outputfileLession)),'w');
    fileID = fopen(char(fullfile(resultf, outputfileLession)),'w');
    fprintf(fileID,'%s, %s, %s, %s, %s \n', 'fname', 'symptom', ...
        'area', 'eccentricity', 'orientation');
    
    for i=1:length(rd)
        name0 = rd(i).name;
        char3 =  strread(name0,'%s','delimiter','.');
        %fname = strcat('..\', imagefolder, '\', name0)
        fname = fullfile(imagefolder, name0)
        I = imread(fname);
        
        %% if relevant get illuminant corrected image
        imshow(I)
        imCorrection = input('Get illuminant corrected image? (y or n): ', 's');
        if strcmp(imCorrection, 'y')
            I = illuminant_correction(I);
        end
        
        close all
        %% Select ROI
        CI = cropImage(I);
        %Segment image
        BW = classifyImage(CI);
        
        % Plot results
        saveimage(I, BWB);
        % extract features
        [ne] = extractFeatures(BWB);
        % save features in file
        savedata(fileID, ne, char3(1));
        close all;
     
    end
    fclose(fileID)
  
    
%Classify image by k-means - unsupervised  
% CI = Cropped image
% Return BW, disease symptoms
function[BW] = classifyImage(CI)
    cform = makecform('srgb2lab');
    lab_he = applycform(CI,cform);
    ab = double(lab_he(:,:,2:3));
    nrows = size(ab,1);
    ncols = size(ab,2);
    ab = reshape(ab,nrows*ncols,2);

    nColors = 3;
    % repeat the clustering 3 times to avoid local minima
    [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
                                  pixel_labels = reshape(cluster_idx,nrows,ncols);
    imshow(pixel_labels,[]), title('image labeled by cluster index')
    segmented_images = cell(1,3);
    rgb_label = repmat(pixel_labels,[1 1 3]);

    for k = 1:nColors
        color = CI;
        color(rgb_label ~= k) = 0;
        segmented_images{k} = color;
    end
    
    ui = plotClasses(CI, segmented_images);
    X = zeros(size(pixel_labels));
    BW = pixel_labels == ui;
    [fv] = extractFeatures(BW);
   
    
    
%Plot cluster classification
% I = Cropped image
% segmented_images = Clusters
% Return an index indicating which cluster represents a disease
 function[useImage] = plotClasses(I, segmented_images)
    subplot(2,3,2), imshow(I), title('Original Image')
    subplot(2,3,4), imshow(segmented_images{1}), title('objects in cluster 1');
    subplot(2,3,5), imshow(segmented_images{2}), title('objects in cluster 2');
    subplot(2,3,6), imshow(segmented_images{3}), title('objects in cluster 3');
    useImage = input('Select b 1, 2 or 3 (or 0 for orignal');
    
    
function[features] = extractLeafFeafures(leaf, I, fname, tipr, tipg, tipb)
    leaf = imfill(leaf, 'holes');
    r = I(:, :, 1);             % red channel
    g = I(:, :, 2);             % green channel
    b = I(:, :, 3);             % blue channel
    HSV = rgb2hsv(I);
    h = HSV(:, :, 1);             % red channel
    s = HSV(:, :, 2);             % red channel
    v = HSV(:, :, 3);             % red channel
    f = regionprops(leaf, 'all');
    m = find([f.Area] ==  max([f(:).Area]))
    features = f(m);
    BW = roicolor(g*0.5, 65,82);
    %imagesc(BW);
    i = BW == 1;
    features.meanr = mean(r(i));
    features.meang = mean(g(i));
    features.meanb = mean(b(i));
    features.meanh = mean(h(i));
    features.means = mean(s(i));
    features.meanv = mean(v(i));
    features.length = features.BoundingBox(3);
    features.width = features.BoundingBox(4);
    features.tr = tipr;
    features.tg = tipg;
    features.tb = tipb;
    features.fname = fname;
  
    
function createFileHead(outputfile, featureStruct)
    fields = fieldnames(featureStruct);
    c = {'Centroid', 'BoundingBox', 'SubarrayIdx','ConvexHull', 'ConvexImage', 'Image', 'FilledImage' ...
        'Extrema', 'Solidity','PixelIdxList', 'PixelList'};
    i = find(ismember(fields,c));
    fieldsprune = fields;
    fieldsprune(i) = [];
    str='%s,';
    nItem=numel(fieldsprune);
    strAll=repmat(str,1,nItem-2);
    strAll = [strAll, '%s\n'];
    fprintf(outputfile, '%s, ',fieldsprune{nItem:nItem});
    fprintf(outputfile, strAll(1:end),fieldsprune{1:nItem-1});
    
    
function saveFeatureData(outputfile, featureStruct)
    fields = fieldnames(featureStruct);
    c = {'Centroid', 'BoundingBox', 'SubarrayIdx','ConvexHull', 'ConvexImage', 'Image', 'FilledImage' ...
        'Extrema', 'Solidity','PixelIdxList', 'PixelList'};
    i = find(ismember(fields,c));
    %fieldsprune = fields([1,5:8,11,14,15,17,19,22,23,24,25,26, 27,28,29, 30,31]);
    fieldsprune = fields;
    fieldsprune(i) = [];
    nItem=numel(fieldsprune);
    dp = featureStruct.(fieldsprune{nItem});
    fprintf(outputfile, '%s,', dp{1});
    for i=1:nItem-1
        dp = featureStruct.(fieldsprune{i});
        str = '%12.2f,';
        fprintf(outputfile, str, dp);
    end
    fprintf(outputfile, '\n');
           
    
% select background
function[BWB, X, LEAF] = selectBackground(I, BWB)
    X = zeros(size(BWB));
    i = BWB == 0;
    X(i) = 1;
    X = bwareaopen(X, 1000);
    se = strel('disk',1);
    X = imerode(X, se);
    LEAF = X;
    Y = X;
    X = bsxfun(@times, I, cast(X,class(I)));
    i = Y == 1;
    j = Y == 0;
    Y(i) = 0;
    Y(j) = 1;
    se = strel('disk',5);
    BWB = imdilate(Y, se);
  


function[a,b] = StartLeaf(BW, h)
    [B,L] = bwboundaries(BW,'noholes');
    a = B{1}(1,1);
    b = B{1}(1,2);
      

% Show original and segmented images in a plot
function saveimage(I,BWM)
    name='a'
    m = I;
    m(find(BWM)) = 255;
    f=figure('Visible','on');
    labTransformation = makecform('srgb2lab');
    ISEG = applycform(I, labTransformation);
    subplot(3,1,1),imshow(I), title(strcat('Original image',name));
    subplot(3,1,2),imshow(BWM), title('Segment diseased area');;
    subplot(3,1,3),imshow(m), title('Final image');
    saveas(f, char(strcat(name, '_plot', '.png')));

    
%Segment foreground  
% I = Original image
% channel = Colour channel
% mn = min pixel value
% mx = max pixel value
function[BW] = processImage(I, channel, mn, mx)
%     labTransformation = makecform('srgb2lab');
%     ISEG = applycform(I, labTransformation);
%     [counts,x] = imhist(ISEG(:,:,1));
    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);
    BW = roicolor(R, mn, mx);
    BW = bwareaopen(BW, 60);
    
        

%Segment chlorosis
 function[BW] = processChlorosis(I, lid)
    r = I(:, :, 1);             % red channel
    g = I(:, :, 2);             % green channel
    b = I(:, :, 3); 
    h = imhist(r);
    if(lid > 0)
        BW = roicolor(r*3, 255, 255);
        se = strel('disk',1);
        BW = imerode(BW, se);
        BW = bwareaopen(BW, 100);
    else
        BW = zeros(size(I));
    end
    
 %Segment chlorosis
 function[BW] = processNecrosis(I, BWChlorosis, BWB, lid)
    r = I(:, :, 1);             % red channel
    g = I(:, :, 2);             % green channel
    b = I(:, :, 3);
    greeness = double(g) - max(double(r*0.8), double(b));
    o = min(reshape( greeness.' ,1,numel(greeness)))
    m = max(reshape( greeness.' ,1,numel(greeness)))
    %if(size(lid,2) == 0)
    %    SS = searchStripe(I, o, m);
    %end
        
    %figure
    %subplot(1,2,1),imagesc(g);
    %subplot(1,2,2),imagesc(greeness);
    if (o < 0)
        BW = roicolor(greeness, -15, 60);
    elseif (o >= 0 & m <= 110)
        BW = roicolor(greeness, 0, 45);
    else
        BW = roicolor(greeness, 0, 80);
    end
    BW(find(BWB)) = 0;
    se = strel('disk',4);
    BW = imdilate(BW, se);
    BW = imfill(BW,'holes');
    %BW(find(SS)) = 0;
    %BW = deleteFP(BW);

    
function[SS] = searchStripe(I, o, m)
    r = I(:, :, 1);             % red channel
    g = I(:, :, 2);             % green channel
    b = I(:, :, 3);
    if (o < 0)
        SS = roicolor(r, 94, 129);
    elseif (o >= 0 & m <= 110)
        SS = roicolor(b, 96, 140);
    else
         SS = roicolor(b, 85, 140);
    end    
   
    
function[mn, mx] = searchHistogram(I)
    h= imhist(I*1.4);
    if(sum(h(2:150)) < 16000)
        mn = 8; mx=130
    else
        mn = 8; mx=160
    end
        
    
%Merge images
% BWF = Foreground
% BWB = Background
function[m, BWF] = mergeImage(I, BWB, BWN)
    BWF = BWN;
    BWF(find(BWB)) = 0;
    m = I;
    m(find(BWF)) = 255;
    %imshow(m);
    
% Extract features from segmented areas      
function[fv] = extractFeatures(BWF)
    
    fv = regionprops(BWF, 'Area', 'Eccentricity', 'Orientation', 'PixelIdxList');
        
        
    
% Get properties from segmented regions
% BWH = Segmented image
function[a] = filterDataAll(BWH)
    cc = regionprops(BWH, 'Area');
    a = sum(cat(1, cc.Area))
    
% Save data   
% outputfile = File where data are saved
% feature_array = Array with segmented regions
% fname = Image name
function savedata(outputfile, feature_array, fname)
   fname
   for i=1:length(feature_array)
        fprintf(outputfile, '%s, %12.0f, %12.0f, %12.0f \n', fname{1},...
        feature_array(i).Area, feature_array(i).Eccentricity, feature_array(i).Orientation)
    end

       
%Plot for debugging
function plotTrans(I, mchl, mnec, name, result)
    f=figure('Visible','off', 'Position', [100 100 1000 900]);
    subplot(3,1,1), imshow(I);
    subplot(3,1,2), imshow(mchl);
    subplot(3,1,3), imshow(mnec);
    title(strcat('Original image' ,name));
    saveas(f, char(strcat(result, name, '_plot', '.png')));
 
function[BW] = deleteFP(BW)
    %imagesc(bwlabel(BW));
    CC = regionprops(BW, 'Area', 'Eccentricity', 'PixelIdxList', 'Extent', 'Solidity');
    i = (([CC.Area]) > 300 & ([CC.Area]) < 20000 ... 
        & ([CC.Eccentricity]) > 0.99 & ([CC.Extent]) <= 0.50) ...
        | (([CC.Solidity]) >= 0.8 & ([CC.Area]) < 14000 & ([CC.Eccentricity]) > 0.98);
    idx = find(i == 1);
    for i=1:length(idx)
        CC(idx(i))
        BW(CC(idx(i)).PixelIdxList) = false;
    end
    i;
    
    
function[] = searchChlorosis(BW, I)
    y = regionprops(BW, 'Area', 'Eccentricity', 'PixelIdxList');
    [biggestSize,idx] = max(cat(y.Area));
    i = ([y.Area]) > 1500 & ([y.Area]) < 4000 & ([y.Eccentricity]) > 0.9
    idx = find(i == 1);
    BW2 = false(size(BW));
    BW2(y(idx).PixelIdxList) = true;
    

function[maskedRgbImage] = cropImage(I)
        imshow(I);
        BW = roipoly;
        maskedRgbImage = bsxfun(@times, I, cast(BW,class(I)));
        % Display it.
        imshow(maskedRgbImage);

