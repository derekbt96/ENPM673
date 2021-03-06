clc

vid = VideoReader('detectbuoy.avi');

% writerObj = VideoWriter('buoy_mask_new.avi');
% writerObj.FrameRate = 20;
% open(writerObj);


% wnd = 25;
% thres = [80 80 40];

frame_indx = 0;

image_points = [];
while hasFrame(vid)
    frame = readFrame(vid);
    frame_indx = frame_indx+1
%     if frame_indx > 135
%         wnd = 40;
%     end
    imshow(frame)
%     mask = zeros(480,640,3);
    
    [x,y] = ginput(1);
    x1 = round(x);
    y1 = round(y);
%     y_indx = max(y-wnd,0):min(y+wnd,480);
%     x_indx = max(x-wnd,0):min(x+wnd,640);
% 
%     mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(1);
%     mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(1);
%     mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(1);
%     mask(y_indx,x_indx,1) = mask1 & mask2 & mask3;
    

    if frame_indx < 150 || frame_indx > 171
        [x,y] = ginput(1);
        x2 = round(x);
        y2 = round(y);
%         
%         y_indx = max(y-wnd,0):min(y+wnd,480);
%         x_indx = max(x-wnd,0):min(x+wnd,640);
% 
%         mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(2);
%         mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(2);
%         mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(2);
%         mask(y_indx,x_indx,2) = mask1 & mask2 & mask3;
    else
        x2 = 0;
        y2 = 0;
    end

    if frame_indx < 45
        [x,y] = ginput(1);
        x3 = round(x);
        y3 = round(y);
%         y_indx = max(y-wnd,0):min(y+wnd,480);
%         x_indx = max(x-wnd,0):min(x+wnd,640);
% 
%         mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(3);
%         mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(3);
%         mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(3);
%         mask(y_indx,x_indx,3) = mask1 & mask2 & mask3;
    else
        x3 = 0;
        y3 = 0;
    end
%     [x1 y1 x2 y2 x3 y3]
    image_points = [image_points; x1 y1 x2 y2 x3 y3];
%     imshow(mask);
%     writeVideo(writerObj, mask);
    
%     break
end
% close(writerObj);


%%
%{
clc
vid2 = VideoReader('detectbuoy.avi');
vid = VideoReader('buoy_mask.avi');


writerObj = VideoWriter('buoy_mask_new.avi');
writerObj.FrameRate = 20;
open(writerObj);

frame_indx = 0;
wnd = 25;
thres = [80 80 40];

frame = readFrame(vid2);
frame_indx = frame_indx+1;


imshow(frame);
mask = zeros(480,640,3);

[x,y] = ginput(1);
x = round(x);
y = round(y);
y_indx = max(y-wnd,0):min(y+wnd,480);
x_indx = max(x-wnd,0):min(x+wnd,640);

mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(1);
mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(1);
mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(1);
mask(y_indx,x_indx,1) = mask1 & mask2 & mask3;

[x,y] = ginput(1);
x = round(x);
y = round(y);
y_indx = max(y-wnd,0):min(y+wnd,480);
x_indx = max(x-wnd,0):min(x+wnd,640);

mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(2);
mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(2);
mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(2);
mask(y_indx,x_indx,2) = mask1 & mask2 & mask3;       

[x,y] = ginput(1);
x = round(x);
y = round(y);
y_indx = max(y-wnd,0):min(y+wnd,480);
x_indx = max(x-wnd,0):min(x+wnd,640);

mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(3);
mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(3);
mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(3);
mask(y_indx,x_indx,3) = mask1 & mask2 & mask3;

imshow(mask);
writeVideo(writerObj, mask);






while hasFrame(vid)
    frame = readFrame(vid);
    frame_indx = frame_indx+1
    
    mask = zeros(480,640,3);
    mask(:,:,1) = frame(:,1:640,1) > 1;
    mask(:,:,2) = frame(:,1:640,2) > 1;
    mask(:,:,3) = frame(:,1:640,3) > 1;
    writeVideo(writerObj, mask);

end
close(writerObj);

%%

vid = VideoReader('detectbuoy.avi');

% writerObj = VideoWriter('buoy_mask_new2.avi');
% writerObj.FrameRate = 20;
% open(writerObj);

frame_indx = 0;
while hasFrame(vid)
    frame = readFrame(vid);
    frame_indx = frame_indx+1
    imshow(frame);

    if frame_indx == 150
        
        frame_indx = frame_indx
    end
        
%     writeVideo(writerObj, frame);
end
% close(writerObj);

% 1: 
% 2: <150, >171
% 3: <45, 
%}