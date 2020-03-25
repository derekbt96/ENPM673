clc

vid = VideoReader('detectbuoy.avi');

% writerObj = VideoWriter('buoy_mask_new.avi');
% writerObj.FrameRate = 20;
% open(writerObj);


wnd = 25;
thres = [80 80 40];

frame_indx = 0;

% frame = readFrame(vid);
% for a = 1:43
%     frame = readFrame(vid);
% end

while hasFrame(vid)
    frame = readFrame(vid);
    frame_indx = frame_indx+1
    
    imshow(frame);
    mask = zeros(480,680,3);
    
    [x,y] = ginput(1);
    x = round(x);
    y = round(y);
    y_indx = max(y-wnd,0):min(y+wnd,480);
    x_indx = max(x-wnd,0):min(x+wnd,640);

    mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(1);
    mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(1);
    mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(1);
    mask(y_indx,x_indx,1) = mask1 & mask2 & mask3;
%         imshow(mask);

    [x,y] = ginput(1);
    x = round(x);
    y = round(y);
    y_indx = max(y-wnd,0):min(y+wnd,480);
    x_indx = max(x-wnd,0):min(x+wnd,640);

    mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(2);
    mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(2);
    mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(2);
    mask(y_indx,x_indx,2) = mask1 & mask2 & mask3;       
%     imshow(mask);

    if frame_indx < 45
        [x,y] = ginput(1);
        x = round(x);
        y = round(y);
        y_indx = max(y-wnd,0):min(y+wnd,480);
        x_indx = max(x-wnd,0):min(x+wnd,640);

        mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(3);
        mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(3);
        mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(3);
        mask(y_indx,x_indx,3) = mask1 & mask2 & mask3;
    end
        
    imshow(mask);
    writeVideo(writerObj, mask);
    
%     break
    
    
end
close(writerObj);


%%
vid2 = VideoReader('detectbuoy.avi');
vid = VideoReader('detectbuoy.avi');


writerObj = VideoWriter('buoy_mask_new.avi');
writerObj.FrameRate = 10;
open(writerObj);

frame_indx = 0;

frame = readFrame(vid2);
frame_indx = frame_indx+1;


imshow(frame);
mask = zeros(480,680,3);

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

if frame_indx < 45
    [x,y] = ginput(1);
    x = round(x);
    y = round(y);
    y_indx = max(y-wnd,0):min(y+wnd,480);
    x_indx = max(x-wnd,0):min(x+wnd,640);

    mask1 = abs(single(frame(y_indx,x_indx,1))-single(frame(y,x,1))) < thres(3);
    mask2 = abs(single(frame(y_indx,x_indx,2))-single(frame(y,x,2))) < thres(3);
    mask3 = abs(single(frame(y_indx,x_indx,3))-single(frame(y,x,3))) < thres(3);
    mask(y_indx,x_indx,3) = mask1 & mask2 & mask3;
end

imshow(mask);
writeVideo(writerObj, mask);






while hasFrame(vid)
    frame = readFrame(vid);
    frame_indx = frame_indx+1
    
    writeVideo(writerObj, frame);

end
close(writerObj);