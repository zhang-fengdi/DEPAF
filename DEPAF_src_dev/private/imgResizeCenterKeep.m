function IResized = imgResizeCenterKeep(I, scale, interpMethod)
% imgResizeCenterKeep obtains a center-aligned scaled image based on interp3.
%
%  This function scales a given image while keeping its center aligned. It uses the interp3 method
%  to interpolate and produce the scaled image. This approach is suitable for scenarios where
%  center alignment is needed during image scaling.
%
%  Input Parameters:
%    I - Original image.
%    scale - Scaling factors represented as a three-element vector [x-axis scale, y-axis scale, z-axis scale].
%    interpMethod - Interpolation method, such as 'spline', 'nearest', 'linear', 'cubic'.
%
%  Output Parameters:
%    IResized - Scaled image.

% Get the original coordinate range:
[oriWid, oriLen, oriHt] = size(I);
YRange = 1:oriWid;
XRange = 1:oriLen;
ZRange = 1:oriHt;

% Calculate dimensions of the scaled image:
if scale(1) >= 1
    tarWid = ceil(oriWid * scale(1));
else
    tarWid = floor(oriWid * scale(1));
end
if scale(2) >= 1
    tarLen = ceil(oriLen * scale(2));
else
    tarLen = floor(oriLen * scale(2));
end
if scale(3) >= 1
    tarHt = ceil(oriHt * scale(3));
else
    tarHt = floor(oriHt * scale(3));
end

% Obtain query coordinates for scaling:
YqStepSize = 1 / scale(1);
XqStepSize = 1 / scale(2);
ZqStepSize = 1 / scale(3);
YqRange = 1 : YqStepSize : YqStepSize*(tarWid+scale(1)-1);
XqRange = 1 : XqStepSize : XqStepSize*(tarLen+scale(2)-1);
ZqRange = 1 : ZqStepSize : ZqStepSize*(tarHt+scale(3)-1);

% Calculate alignment points:
XAnchor = mean(XRange);
YAnchor = mean(YRange);
ZAnchor = mean(ZRange);
XqAnchor = mean(XqRange);
YqAnchor = mean(YqRange);
ZqAnchor = mean(ZqRange);

% Align coordinates:
XqRange = XqRange + (XAnchor - XqAnchor);
YqRange = YqRange + (YAnchor - YqAnchor);
ZqRange = ZqRange + (ZAnchor - ZqAnchor);

% Convert format to save memory:
XqRange = single(XqRange);
YqRange = single(YqRange);
ZqRange = single(ZqRange);

% Obtain scaled image via interpolation (pad I by 1 to prevent interpolation overflow beyond image boundaries):
I = padarray(I, [1 1 1], 'replicate', 'both');
[Xq, Yq, Zq] = meshgrid(XqRange+1, YqRange+1, ZqRange+1);
IResized = interp3(I, Xq, Yq, Zq, interpMethod);
end