% This program segments visual plant disease symptoms
% Anyela Camargo, June 2014.

function main()
    %change these locations as needed
    rootname =  'C:\Anyela\matlab\brachydisease\images\';
    resultf =  'C:\Anyela\matlab\brachydisease\results\';
    outputfileLession = 'lession.csv';
    outputfileLeaf = 'leaf1.csv';
    %DiseaseFeatures(rootname, outputfileLession, resultf);
    LeafFeatures(rootname, outputfileLeaf, resultf);


% Extract leaf features  
function LeafFeatures(rootname, outputfileLeaf, resultf) 
    
    % ROI by pot in six pots tray. Two rows and three columns.
    % Middle pot first row is empty

    rd = dir(strcat(rootname, 'BD_01300_3*.jpg'));
    fileIDLeaf = fopen(char(strcat(resultf, outputfileLeaf)),'w');
    
    for i=1:length(rd)
        
        name0 = rd(i).name;
        char3 =  strread(name0,'%s','delimiter','.');
        fname = strcat(rootname, name0)
        I = imread(fname);
        %Select background
        BWB = processImage(I, 1, 240, 255); 
        [BWB, IC, leaf] = selectBackground(I, BWB);
        [tr, tg, tb] = selectLeaftip(leaf, I);
        featureStruct = extractLeafFeafures(leaf, I, char3(1), tr, tg, tb);
        if(i == 1)
            createFileHead(fileIDLeaf, featureStruct);
        end
        saveFeatureData(fileIDLeaf, featureStruct)
     close all;
    end
    fclose(fileIDLeaf);
   
 
 
 % Extract disease features
 function DiseaseFeatures(rootname, outputfileLession, resultf)
    rd = dir(strcat(rootname, 'BD*.jpg'));
    fileID = fopen(char(strcat(resultf,outputfileLession)),'w');
    fprintf(fileID,'%s, %s, %s, %s, %s, %s, %s, %s, %s\n', 'fname', 'symptom', ...
        'area', 'eccentricity', 'orientation', 'meanr', 'meang', 'meanb', 'dist2base');
    x = {'BD_01396_1.jpg','BD_01396_2.jpg','BD_01396_3.jpg','BD_01396_4.jpg'};
    
    for i=1:length(rd)
        name0 = rd(i).name;
        char3 =  strread(name0,'%s','delimiter','.');
        fname = strcat(rootname, name0)
        I = imread(fname);
        %Select background
        BWB = processImage(I, 1, 240, 255); 
        [BWB, IC, leaf] = selectBackground(I, BWB);
        %createTraing(I);
        BL = selectLeaf(IC, char3(1));
        [a,b] = StartLeaf(leaf);
        %imshow(BL);
        lid = find(ismember(x,name0));
        BWChlorosis = processChlorosis(IC, lid);
        BWNecrosis = processNecrosis(IC, BWChlorosis, BWB, lid);
        %BWNecrosis(find(BL)) = 0;
        %figure
        %subplot(3,1,1), imshow(I); subplot(3,1,2), imshow(BWChlorosis); 
        %subplot(3,1,3), imshow(BWNecrosis);
        [mchl, BM] = mergeImage(I, BWB, BWChlorosis);
        [chl] = extractFeatures(I, BM, a,b);
        [mnec, BM] = mergeImage(I, BWB, BWNecrosis);
        
        [ne] = extractFeatures(I, BM,a,b);
        %figure
        %subplot(3,1,1), imshow(I); subplot(3,1,2), imshow(mchl); 
        %subplot(3,1,3), imshow(mnec)
        
        plotTrans(I, mchl, mnec, char3(1), resultf);
        % plot image
        if (size(lid,2) > 0)
            savedata(fileID, chl, char3(1), 'Chlorosis' );
        end
        savedata(fileID, ne, char3(1), 'Necrosis');
        close all;
     
    end
    fclose(fileID)
  
    
    
function createTraining(I)
    imshow(I);
    BIC = imcrop;
    

function[BL] = selectLeaf(I, name)
    r = I(:, :, 1);             % red channel
    g = I(:, :, 2);             % green channel
    b = I(:, :, 3);             % blue channel
    greeness = double(g) - max(double(r), double(b));
    %f=figure
    %subplot(2,1,1), imshow(I);
    %subplot(2,1,2), imhist(g);
    h = imhist(g);
    max(h(2:240))
    %saveas(f, char(strcat(name, '_plot', '.png')));
    BL = roicolor(greeness, 54, 116);
    
    
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
function saveimage(I,BWM, BWB,BWL,BWF, name)
    m = I;
    m(find(BWM)) = 255;
    f=figure('Visible','on');
    labTransformation = makecform('srgb2lab');
    ISEG = applycform(I, labTransformation);
    subplot(3,2,1),imshow(I), title(strcat('Original image',name));
    subplot(3,2,2),imhist(ISEG(:,:,1)), title('Hist');
    subplot(3,2,3),imshow(BWB), title('Segment background');
    subplot(3,2,4),imshow(BWL), title('Segment leaf');
    subplot(3,2,5),imshow(BWF), title('Segment diseased area');;
    subplot(3,2,6),imshow(m), title('Final image');
    saveas(f, char(strcat(name, '_plot', '.png')));

    
%Segment foreground  
% I = Original image
% channel = Colour channel
% mn = min pixel value
% mx = max pixel value
function[BW] = processImage(I, channel, mn, mx)
    labTransformation = makecform('srgb2lab');
    ISEG = applycform(I, labTransformation);
    [counts,x] = imhist(ISEG(:,:,1));
    BW = roicolor(ISEG(:,:,channel), mn, mx);
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
function[fv] = extractFeatures(I, BWF, a,b)
    
    fv = regionprops(BWF, 'Area', 'Eccentricity', 'Orientation', 'PixelIdxList');
    for i=1:length(fv)
       
       u = I(:,:,1);
       fv(i).meanr = mean(u(fv(i).PixelIdxList));
       u = I(:,:,2);
       fv(i).meang = mean(u(fv(i).PixelIdxList));
       u = I(:,:,3);
       fv(i).meanb = mean(u(fv(i).PixelIdxList));
       U = zeros(size(u));
       U(fv(i).PixelIdxList) = 1;
       [a1, b1] = StartLeaf(U);
       fv(i).dist2base = sqrt((a-a1)^2+(b-b1)^2);
    end
        
        
    
% Get properties from segmented regions
% BWH = Segmented image
function[a] = filterDataAll(BWH)
    cc = regionprops(BWH, 'Area');
    a = sum(cat(1, cc.Area))
    
% Save data   
% outputfile = File where data are saved
% feature_array = Array with segmented regions
% fname = Image name
function savedata(outputfile, feature_array, fname, lessionname)
   
   for i=1:length(feature_array)
        fprintf(outputfile, '%s, %s, %12.0f, %12.0f, %12.0f, %12.0f,%12.0f,%12.0f, %12.0f \n', fname{1}, ...
            lessionname,...
        feature_array(i).Area, feature_array(i).Eccentricity, feature_array(i).Orientation, ... 
        feature_array(i).meanr, feature_array(i).meang, feature_array(i).meanb,...
        feature_array(i).dist2base);
    end

       
%Plot for debugging
function plotTrans(I, mchl, mnec, name, result)
    f=figure('Visible','off', 'Position', [100 100 1000 900]);
    subplot(3,1,1), imshow(I);
    subplot(3,1,2), imshow(mchl);
    subplot(3,1,3), imshow(mnec);
    title(strcat('Original image' ,name));
    saveas(f, char(strcat(result, name, '_plot', '.png')));
 
    
function[tr,tg, tb] = selectLeaftip(leaf, I) 
    leaf = bwareaopen(leaf, 10000);
    [B,L] = bwboundaries(leaf,'noholes');
    i = find(B{1}(:,2) == max(B{1}(:,2)));
    i = max(i);
    x = 30;
    I2 = imcrop(I,[(B{1}(i,2)-x) (B{1}(i,1)-10) x+12 20]);
    labTransformation = makecform('srgb2lab');
    ISEG = applycform(I2, labTransformation);
    %imagesc(ISEG(:,:,2));
    r = I2(:, :, 1);             % red channel
    g = I2(:, :, 2);             % green channel
    b = I2(:, :, 3);  
    tr = mean(r(find(r ~= 255)));
    tg = mean(g(find(g ~= 255)));
    tb = mean(b(find(b ~= 255)));
    
 
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
    
    
    