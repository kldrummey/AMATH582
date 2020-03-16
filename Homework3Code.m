%% AMATH582 - Homework 3

%% Ideal Case - Entire motion is in the z direction. camN_1.mat, where N=1,2,3
close all; clear all; clc;

load cam1_1.mat; load cam2_1.mat; load cam3_1.mat

%Convert variables to videos to visually inspect paint can trajectory
v1_1=VideoWriter('vid1_1.avi'); open(v1_1); writeVideo(v1_1,vidFrames1_1); close(v1_1);
v2_1=VideoWriter('vid2_1.avi'); open(v2_1); writeVideo(v2_1,vidFrames2_1); close(v2_1);
v3_1=VideoWriter('vid3_1.avi'); open(v3_1); writeVideo(v3_1,vidFrames3_1); close(v3_1);

%Convert uint8 to double, determine shortest video to set frame limit
vid11=double(vidFrames1_1); vid21=double(vidFrames2_1); vid31=double(vidFrames3_1);
vidsize1=size(vid11); vidsize2=size(vid21); vidsize3=size(vid31);
vidn1lengths=[size(vid11,4); size(vid21,4); size(vid31,4)];
framesn1=1:min(vidn1lengths);

%Set up X_n and Y_n matrices and assign values based on brightest pixels
X1=zeros(1, length(framesn1)); X2=X1; X3=X1; Y1=X1; Y2=X1; Y3=X1; 

for j=1:length(X1)
    v11gray=rgb2gray(vidFrames1_1(:,:,:,j));
    [x1 y1]=find(v11gray> 240);
    X1(j)=mean(x1); Y1(j)=mean(y1);
    v21gray=rgb2gray(vidFrames2_1(:,:,:,j));
    [x2 y2]=find(v21gray>240);
    X2(j)=mean(x2); Y2(j)=mean(y2);
    v31gray=rgb2gray(vidFrames3_1(:,:,:,j));
    [x3 y3]=find(v31gray>240);
    X3(j)=mean(x3); Y3(j)=mean(y3);
end

%Plot X-Y trajectory of the paint can in each video
figure(1)
subplot(1,3,1); plot3(X1,Y1,framesn1); title('Camera 1','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
subplot(1,3,2); plot3(X2,Y2,framesn1); title('Camera 2','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
subplot(1,3,3); plot3(X3,Y3,framesn1); title('Camera 3','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
sgtitle('Raw trajectory data, Video Set 1','Fontsize',[16]);

%SVD analysis on trajectories
n1_matrix=[X1;X2;X3;Y1;Y2;Y3];
[a1,b1]=size(n1_matrix);
ab=mean(n1_matrix,2);
n1_matrix=n1_matrix-repmat(ab,1,b1);
[u1,s1,v1]=svd(n1_matrix/sqrt(n1_matrix-b1)); %SVD
lambda=diag(s1).^2; %diagonal variances
proj1=u1'*n1_matrix;

figure(2)
plot(lambda/sum(lambda)*100,'ko','MarkerSize',[10],'MarkerFaceColor','k'); xlim([0 6.5]); 
title('Variance captured by each mode, Videos in Set 1','Fontsize',[16]);
ylabel('Percent of the variance','Fontsize',[16]); xlabel('Mode','Fontsize',[16]);

figure(3)
for j=1:6
    hold on
    plot(framesn1,proj1(j,:),'LineWidth',2)
    title('Paint can displacement by mode')
    legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6'); xlim([0 length(framesn1)]);
end
xlabel('Frame','Fontsize',[16]); 

figure(4)
for j=1:2
    subplot(2,1,j)
    plot(framesn1,proj1(j,:),'LineWidth',2)
    title(sprintf('Paint can displacement, mode %d',j)); xlim([0 length(framesn1)]);
end
xlabel('Frame','Fontsize',[16])
%% Noisy Case - Camera is shaking. camN_2.mat, where N=1,2,3
close all; clear all; clc;

load cam1_2.mat; load cam2_2.mat; load cam3_2.mat

%Convert variables to videos to visually inspect paint can trajectory
v1_2=VideoWriter('vid1_2.avi'); open(v1_2); writeVideo(v1_2,vidFrames1_2); close(v1_2);
v2_2=VideoWriter('vid2_2.avi'); open(v2_2); writeVideo(v2_2,vidFrames2_2); close(v2_2);
v3_2=VideoWriter('vid3_2.avi'); open(v3_2); writeVideo(v3_2,vidFrames3_2); close(v3_2);

%Convert uint8 to double, determine shortest video to set frame limit
vid12=double(vidFrames1_2); vid22=double(vidFrames2_2); vid32=double(vidFrames3_2);
vidsize12=size(vid12); vidsize22=size(vid22); vidsize32=size(vid32);
vidn2lengths=[size(vid12,4); size(vid22,4); size(vid32,4)];
framesn2=1:min(vidn2lengths);

%Set up X_n and Y_n matrices and assign values based on brightest pixels
X12=zeros(1, length(framesn2)); X22=X12; X32=X12; Y12=X12; Y22=X12; Y32=X12; 

for j=1:length(X12)
    v12gray=rgb2gray(vidFrames1_2(:,:,:,j));
    [x12 y12]=find(v12gray> 240);
    X12(j)=mean(x12); Y12(j)=mean(y12);
    v22gray=rgb2gray(vidFrames2_2(:,:,:,j));
    [x22 y22]=find(v22gray>240);
    X22(j)=mean(x22); Y22(j)=mean(y22);
    v32gray=rgb2gray(vidFrames3_2(:,:,:,j));
    [x32 y32]=find(v32gray>240);
    X32(j)=mean(x32); Y32(j)=mean(y32);
end

%Plot X-Y trajectory of the paint can in each video
figure(1)
subplot(1,3,1); plot3(X12,Y12,framesn2); title('Camera 1','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
subplot(1,3,2); plot3(X22,Y22,framesn2); title('Camera 2','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
subplot(1,3,3); plot3(X32,Y32,framesn2); title('Camera 3','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
sgtitle('Raw Trajectory Data, Video Set 2','Fontsize',[16])

%SVD analysis on trajectories
n2_matrix=[X12;X22;X32;Y12;Y22;Y32];
[a2,b2]=size(n2_matrix);
ab2=mean(n2_matrix,2);
n2_matrix=n2_matrix-repmat(ab2,1,b2);
[u2,s2,v2]=svd(n2_matrix/sqrt(n2_matrix-b2)); %SVD
lambda2=diag(s2).^2; %diagonal variances
proj2=u2'*n2_matrix;

figure(2)
plot(lambda2/sum(lambda2)*100,'ko','MarkerSize',[10],'MarkerFaceColor','k'); xlim([0 6.5]); 
title('Variances captured by each mode, Videos in Set 2','Fontsize',[16]);
ylabel('Percent of the variance','Fontsize',[16]); xlabel('Mode','Fontsize',[16]);

figure(3)
for j=1:6
    hold on
    plot(framesn2,proj2(j,:),'LineWidth',2)
    title('Paint can displacement by mode, Video Set 2','Fontsize',[14])
    legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6'); xlim([0 length(framesn2)]);
end
xlabel('Frame','Fontsize',[16]); 

figure(4)
for j=1:4
    subplot(4,1,j)
    plot(framesn2,proj2(j,:),'LineWidth',2)
    title(sprintf('Paint can displacement Video Set 2, mode %d',j)); xlim([0 length(framesn2)]);
end
xlabel('Frame','Fontsize',[16])

%% Horizontal Displacement - Motion in x, y, z directions. camN_3.mat, where N=1,2,3

close all; clear all; clc;

load cam1_3.mat; load cam2_3.mat; load cam3_3.mat

%Convert variables to videos to visually inspect paint can trajectory
v1_3=VideoWriter('vid1_1.avi'); open(v1_3); writeVideo(v1_3,vidFrames1_3); close(v1_3);
v2_3=VideoWriter('vid2_1.avi'); open(v2_3); writeVideo(v2_3,vidFrames2_3); close(v2_3);
v3_3=VideoWriter('vid3_1.avi'); open(v3_3); writeVideo(v3_3,vidFrames3_3); close(v3_3);

%Convert uint8 to double, determine shortest video to set frame limit
vid13=double(vidFrames1_3); vid23=double(vidFrames2_3); vid33=double(vidFrames3_3);
vidsize13=size(vid13); vidsize23=size(vid23); vidsize33=size(vid33);
vidn3lengths=[size(vid13,4); size(vid23,4); size(vid33,4)];
framesn3=1:min(vidn3lengths);

%Set up X_n and Y_n matrices and assign values based on brightest pixels
X13=zeros(1, length(framesn3)); X23=X13; X33=X13; Y13=X13; Y23=X13; Y33=X13;

for j=1:length(X13)
    v13gray=rgb2gray(vidFrames1_3(:,:,:,j));
    [x13 y13]=find(v13gray>230);
    X13(j)=mean(x13); Y13(j)=mean(y13); 
    v23gray=rgb2gray(vidFrames2_3(:,:,:,j));
    [x23 y23]=find(v23gray>230);
    X23(j)=mean(x23); Y23(j)=mean(y23);
    v33gray=rgb2gray(vidFrames3_3(:,:,:,j));
    [x33 y33]=find(v33gray>230);
    X33(j)=mean(x33); Y33(j)=mean(y33);
end

%Plot X-Y-Z trajectory of the paint can in each video
figure(1)
subplot(1,3,1); plot3(X13,Y13,framesn3); title('Camera 1','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
subplot(1,3,2); plot3(X23,Y23,framesn3); title('Camera 2','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
subplot(1,3,3); plot3(X33,Y33,framesn3); title('Camera 3','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
sgtitle('Raw trajectory data, Video Set 3')

%SVD analysis on trajectories
n3_matrix=[X13;X23;X33;Y13;Y23;Y33];
[a3,b3]=size(n3_matrix);
ab3=mean(n3_matrix,2);
n3_matrix=n3_matrix-repmat(ab3,1,b3);
[u3,s3,v3]=svd(n3_matrix/sqrt(n3_matrix-b3)); %SVD
lambda3=diag(s3).^2; %diagonal variances
proj3=u3'*n3_matrix;

figure(2)
plot(lambda3/sum(lambda3)*100,'ko','MarkerSize',[10],'MarkerFaceColor','k'); xlim([0 6.5]); 
title('Variances captured by each mode, Videos in Set 3','Fontsize',[16]);
ylabel('Percent of the variance','Fontsize',[16]); xlabel('Mode','Fontsize',[16]);

figure(3)
for j=1:6
    hold on
    plot(framesn3,proj3(j,:),'LineWidth',2)
    title('Paint can displacement by mode, Video Set 3','Fontsize',[14])
    legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6'); xlim([0 length(framesn3)]);
end
xlabel('Frame','Fontsize',[16]); 

figure(4)
for j=1:6
    subplot(6,1,j)
    plot(framesn3,proj3(j,:),'LineWidth',2)
    title(sprintf('Paint can displacement Video Set 3, mode %d',j)); xlim([0 length(framesn3)]);
end
xlabel('Frame','Fontsize',[16])
%% Horizontal Displacement/Rotation - Pendulum motion and oscillation in z. camN_4.mat, where N = 1,2,3

close all; clear all; clc;

%Load .mat files with video data
load cam1_4.mat; load cam2_4.mat; load cam3_4.mat

%Convert variables to videos to visually inspect paint can trajectory
v1_4=VideoWriter('vid1_4.avi'); open(v1_4); writeVideo(v1_4,vidFrames1_4); close(v1_4);
v2_4=VideoWriter('vid2_4.avi'); open(v2_4); writeVideo(v2_4,vidFrames2_4); close(v2_4);
v3_4=VideoWriter('vid3_4.avi'); open(v3_4); writeVideo(v3_4,vidFrames3_4); close(v3_4);

%Convert uint8 data to double, determine shortest video to set frame limit
vid14=double(vidFrames1_4); vid24=double(vidFrames2_4); vid34=double(vidFrames3_4);
vidsize14=size(vid14); vidsize24=size(vid24); vidsize33=size(vid34);
vidn4lengths=[size(vid14,4); size(vid24,4); size(vid34,4)];
framesn4=1:min(vidn4lengths);

%Set up X_n and Y_n matrices and assign values based on brightest pixels,
%assuming that converting to grayscale will have pixel values of 0-255,
%with closer to 255 being the brightest.
X14=zeros(1, length(framesn4)); X24=X14; X34=X14; Y14=X14; Y24=X14; Y34=X14; 
for j=1:length(X14)
    v14gray=rgb2gray(vidFrames1_4(:,:,:,j));
    [x14 y14]=find(v14gray> 230);
    X14(j)=mean(x14); Y14(j)=mean(y14);
    v24gray=rgb2gray(vidFrames2_4(:,:,:,j));
    [x24 y24]=find(v24gray>230);
    X24(j)=mean(x24); Y24(j)=mean(y24); 
    v34gray=rgb2gray(vidFrames3_4(:,:,:,j));
    [x34 y34]=find(v34gray>230);
    X34(j)=mean(x34); Y34(j)=mean(y34); 
end

%Plot raw trajectory of the paint can in each video
figure(1)
subplot(1,3,1); plot3(X14,Y14,framesn4); title('Camera 1','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
subplot(1,3,2); plot3(X24,Y24,framesn4); title('Camera 2','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
subplot(1,3,3); plot3(X34,Y34,framesn4); title('Camera 3','Fontsize',[12]); xlabel('X'); ylabel('Y'); zlabel('Frames')
sgtitle('Raw trajectory data, Video Set 4','Fontsize',[16]);

%SVD analysis on trajectories
n4_matrix=[X14;X24;Y14;Y24;X34;Y34];
[a4,b4]=size(n4_matrix);
ab4=mean(n4_matrix,2);
n4_matrix=n4_matrix-repmat(ab4,1,b4);
[u4,s4,v4]=svd(n4_matrix/sqrt(n4_matrix-b4)); %SVD
lambda4=diag(s4).^2; %diagonal variances
proj4=u4'*n4_matrix;

%Plot % of the variance explained by each component
figure(2)
plot(lambda4/sum(lambda4)*100,'ko','MarkerSize',[10],'MarkerFaceColor','k'); xlim([0 6.5]); 
title('Variances explained by each mode, Video Set 4','Fontsize',[16]);
ylabel('Percent of the variance','Fontsize',[16]); xlabel('Mode','Fontsize',[16]);

%Plot modes vs time
figure(3)
for j=1:5
    hold on
    plot(framesn4,proj4(j,:),'LineWidth',2)
    title('Paint can displacement by mode, Video Set 4','Fontsize',[14])
    legend('Mode 1','Mode 2','Mode 3','Mode 4','Mode 5','Mode 6'); xlim([0 length(framesn4)]);
end
xlabel('Frame','Fontsize',[16]); 

figure(4)
for j=1:6
    subplot(6,1,j)
    plot(framesn4,proj4(j,:),'LineWidth',2)
    title(sprintf('Paint can displacement Video Set 4, mode %d',j)); xlim([0 length(framesn4)]);
end
xlabel('Frame','Fontsize',[16])