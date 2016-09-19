function [imGW, imMaxRGB, imMink4] = illuminant_correction(im)

% im -  the RGB image 
% imshow(im)
% [min(im(:)) max(im(:))]
[row,col] = size(im(:,:,1));
im2d = reshape(im, [row*col 3]);
im2d = im2double(im2d);


%Grey World
LGW = mean(im2d);
% illuminant corrected image
imGW = im2double(im);
c=1; imGW(:,:,c) = imGW(:,:,c) ./ LGW(c);
c=2; imGW(:,:,c) = imGW(:,:,c) ./ LGW(c);
c=3; imGW(:,:,c) = imGW(:,:,c) ./ LGW(c);
%imshow(uint8(imGW.*255))

%MaxRGB
LMaxRGB = max(im2d);
% illuminant corrected image
imMaxRGB = im2double(im);
c=1; imMaxRGB(:,:,c) = imMaxRGB(:,:,c) ./ LMaxRGB(c);
c=2; imMaxRGB(:,:,c) = imMaxRGB(:,:,c) ./ LMaxRGB(c);
c=3; imMaxRGB(:,:,c) = imMaxRGB(:,:,c) ./ LMaxRGB(c);
%imshow(uint8(imMaxRGB.*255))


%Lp norm - norm 4
% e = [mu_p(R) mu_p(G) mu_p(B)]
% mu_p(X) = ((sum(X^p)^(1/p))  / (N^(1/p))
pnorm = 4;
%estimate the light based on p-norm
c=1; LMink4(1,c) = (sum(im2d(:,c) .^ pnorm) .^ (1/pnorm)) / ((row*col) ^ (1/pnorm));
c=2; LMink4(1,c) = (sum(im2d(:,c) .^ pnorm) .^ (1/pnorm)) / ((row*col) ^ (1/pnorm));
c=3; LMink4(1,c) = (sum(im2d(:,c) .^ pnorm) .^ (1/pnorm)) / ((row*col) ^ (1/pnorm));
% illuminant corrected image
imMink4 = im2double(im);
c=1; imMink4(:,:,c) = imMink4(:,:,c) ./ LMink4(c);
c=2; imMink4(:,:,c) = imMink4(:,:,c) ./ LMink4(c);
c=3; imMink4(:,:,c) = imMink4(:,:,c) ./ LMink4(c);
%imshow(uint8(imMink4.*255))
