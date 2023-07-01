function []=point_to_image(path,point_set)
close all
%% Camera internal parameters
CAM_WID = 1280; 
CAM_HGT =720;
CAM_FX = 608; 
CAM_FY = 608;
CAM_CX = 639;
CAM_CY = 368;
%fig=figure;
%M(450)=struct('cdata',[],'colormap',[]);
point_sets =point_set;
%% four kinds of modes (note colo)
% pctmap: only points 
% pctreal: point with color 
% pctonlyexpansion: expansion point without color
% pctexpansion: expansion point with color
mode="pctexpansion";
switch mode
    case "pctreal"
    save_dir=[fileparts(path),'\pctreal'];
    case "pctmap"
    save_dir=[fileparts(path),'\pctmap'];
    case "pctonlyexpansion"
    save_dir=[fileparts(path),'\pctonlyexpansion'];
    case "pctexpansion"
    save_dir=[fileparts(path),'\pctexpansion1'];
end
if exist(save_dir,'file')==0 %%判断文件夹是否存在
    mkdir(save_dir); 
end
[~,datetime,~]=fileparts(path);
img_z = zeros(CAM_HGT,CAM_WID,3)*10;
%筛选有信号强度的点 x,y,z,rFilter points with signal strength
i=1;
k=0;
while i<=450
    k=k+1;
point_sett=squeeze(point_sets(i,:,:));
    EPS = 1.0e-16;
    valid = abs(point_sett(6,:).* point_sett(3,:))> EPS & abs(point_sett(5,:))<=5&abs(point_sett(2,:))>=1;
    for spini= 1:128
        [point_sett(2,spini),point_sett(3,spini)]=spin(point_sett(2,spini),point_sett(3,spini));
    end
    
    z=-point_sett(2,valid);%z坐标
    u = int16(point_sett(1,valid) * CAM_FX ./ z + CAM_CX);
    v = int16(point_sett(3,valid) * CAM_FY ./ z + CAM_CY);
    point_sett=point_sett(:,valid);
    
    valid=u(1,:)>=1&u(1,:)<=CAM_WID-1&v(1,:)>=1&v(1,:)<=CAM_HGT-1;
    
    u= u(valid);
    v=v(valid);
    point_sett=point_sett(:,valid);
    img_z=uint8(img_z);
    %% pctrmap: only points 
    if (mode== "pctmap")
        for q=1:length(u)
            vindex=v(q)+1+0;
            uindex=CAM_WID-u(q)+1+0;
            if(uindex>=1&&uindex<=CAM_WID-1&&vindex>=1&&vindex<=CAM_HGT-1)
                img_z(vindex,uindex,2) = 255;
                img_z(vindex,uindex,1) = 255;
                img_z(vindex,uindex,3) = 0;
            end
        end
    end
    %% pctreal: point with color 
    if (mode== "pctreal")
        for q=1:length(u)
            vindex=v(q)+1+0;
            uindex=CAM_WID-u(q)+1+0;
            if(uindex>=1&&uindex<=CAM_WID-1&&vindex>=1&&vindex<=CAM_HGT-1)
                img_z(vindex,uindex,2) = max(img_z(vindex,uindex,2),abs(point_sett(5,q)/10*255));
                img_z(vindex,uindex,1) = max(img_z(vindex,uindex,1),(point_sett(2,q)/7.5*255));
                img_z(vindex,uindex,3)=255;
            end
        end
    end
    %% 做处理运行这些
    if (mode== "pctonlyexpansion"||mode== "pctexpansion"||mode=="exptest")
        for q=1:length(u)
            theta=atan((double(u(q))-CAM_CX)/CAM_FY);
            thetares=1/4/cos(theta);
            righturange=abs((tan(theta)-tan(theta-thetares))*CAM_FX);
            lefturange =abs((tan(theta)-tan(theta+thetares))*CAM_FX);
            pinv=(lefturange+righturange)/128;
            uindex = max(int16(CAM_WID-u(q)+1-lefturange/8),1):min(int16(CAM_WID-u(q)+1+righturange/8),CAM_WID-1);
            vindex = max(int16(v(q)+1-pinv),1): min(int16(v(q)+1+pinv),CAM_HGT-1);    
                    img_z(vindex,uindex,1) = max(img_z(vindex,uindex,1),(point_sett(4,q)/7.5*255));    
                    img_z(vindex,uindex,2) = max(img_z(vindex,uindex,2),abs(point_sett(5,q)/10*255));
                    img_z(vindex,uindex,3) = img_z(vindex,uindex,3)+10;
                    if mode== "pctonlyexpansion"
                        img_z(vindex,uindex,1)=255;
                        img_z(vindex,uindex,2)=255;
                        img_z(vindex,uindex,3)=0;
                    end
        end
    end
    %subplot(2,1,2)
    %imshow(img_z)
    %% Process every two frames, and merge its front and back frames
    % Consider mmwave frames like frame0,frame1,frame2,frame3,frame4,frame5,...,450 
    % process image when frame=1,3,5,7,9,..,449
    % while frame1 = frame0 + frame1 + frame2, frame3 = frame2 + frame3 + frame4
    plan=1; 
    switch plan
        case 0
            if(mod(k,3)==0)
                img_save=img_z;
                %imshow(img_save)
                imwrite(img_save,strcat(save_dir,'\',datetime,num2str(int16(i-2), '%05d'),'.png'),'bitdepth',8);
                img_z = zeros(CAM_HGT,CAM_WID,3);
                i=i-1;
            end
        case 1
    %% Other Choice Process every two frames, and only merge its front
            if(mod(k,2)==0)
                img_save=img_z;
                %imshow(img_save)
                imwrite(img_save,strcat(save_dir,'\',datetime,num2str(int16(i-1), '%05d'),'.png'),'bitdepth',8);
                img_z = zeros(CAM_HGT,CAM_WID,3);
            end
    end
    i=i+1;
end

