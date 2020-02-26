%AMATH582 Winter Quarter 2020
%Homework Assignment #1 - Kristen Drummey

%% ESTABLISH PARAMETERS
clear all; close all; clc;
load Testdata
L=15; % spatial domain
n=64; % Fourier modes

x2=linspace(-L,L,n+1); x=x2(1:n); y=x; z=x; %domain discretization
k=(2*pi/(2*L))*[0:(n/2-1) -n/2:-1]; %frequency components of FFT
ks=fftshift(k); %unshifted frequency components

[X,Y,Z]=meshgrid(x,y,z); %spatial axes
[Kx,Ky,Kz]=meshgrid(ks,ks,ks); %frequency axes

%Reshape data into 64x64x64 matrix with 20 time points.
for j=1:20
   Un(:,:,:)=reshape(Undata(j,:),n,n,n);
   close all, isosurface(X,Y,Z,abs(Un),0.4)
   axis([-20 20 -20 20 -20 20]), grid on, drawnow
   pause(1)
end

%% AVERAGE DATA OVER 20 TRIALS

%Average data over 20 trials. 
ave=zeros(n,n,n); %Setup matrix of zeroes to populate
 
for j=1:20
    Un(:,:,:)=reshape(Undata(j,:),n,n,n);
    Utn=fftn(Un); %3-dimensional FFT on reshaped data
    ave(:,:,:)=ave+Utn(:,:,:);
end

ave=ave/j; %divide populated matrix by the number of trials to determine average
Utns=fftshift(ave); %shift averaged data
Utnnorm=abs(Utns)/max(abs(Utns(:))); %normalize data for plotting by dividing by max value

figure(1)
close all, isosurface(Kx,Ky,Kz,Utnnorm,0.6)
axis([-6 6 -6 6 -6 6]), grid on, drawnow
xlabel('Kx','Fontsize',[16]); ylabel('Ky','Fontsize',[16]); zlabel('Kz','Fontsize',[16])
title('Averaged Frequency Content','Fontsize',[16])

%% CONSTRUCT GAUSSIAN FILTER AND FILTER DATA

%Determine frequency signature of the marble.
[M I]=max(Utns(:));
fx=Kx(I); fy=Ky(I); fz=Kz(I);

%Gaussian filter using frequency signature
filter=exp(-0.2*((Kx-fx).^2 + (Ky-fy).^2 + (Kz-fz).^2));

%Double check that the filter looks correct - Gaussian filter should be
%circular (or in this case, spherical)
figure(2)
close all; isosurface(Kx,Ky,Kz,abs(filter),0.2);
grid on, drawnow
xlabel('Kx','Fontsize',[16]); ylabel('Ky','Fontsize',[16]); zlabel('Kz','Fontsize',[16])
title('Gaussian filter','Fontsize',[16])

%Filter data across 20 trials
for a=1:20
    Un(:,:,:)=reshape(Undata(a,:),n,n,n); %reshape original data
    Ut(:,:,:)=fftn(Un); %apply n-dimensional FFT to data
    Uts(:,:,:)=fftshift(Ut); %shift transformed data
    Utsf(:,:,:)=filter.*Uts; %apply Gaussian filter to transformed data
    Unf(:,:,:)=ifftn(ifftshift(Utsf));  %revert filtered data to spatial domain 
end

%Plot filtered frequency data
figure(3)
close all; isosurface(Kx,Ky,Kz,(abs(Utsf)/max(abs(Utsf(:)))),0.2);
axis([-6 6 -6 6 -6 6]),grid on, drawnow
xlabel('Kx','Fontsize',[16]); ylabel('Ky','Fontsize',[16]); zlabel('Kz','Fontsize',[16])
title('Filtered frequency data','Fontsize',[16])

%Plot filtered spatial data
figure(4)
close all; isosurface(X,Y,Z,(abs(Unf)/max(abs(Unf(:)))),0.2);
axis([-20 20 -20 20 -20 20]),grid on, drawnow
xlabel('X','Fontsize',[16]); ylabel('Y','Fontsize',[16]); zlabel('Z','Fontsize',[16])
title('Filtered spatial data','Fontsize',[16])

%% DETERMINE AND PLOT PATH OF THE MARBLE
%Plot path of marble
xcoord=zeros(20,1); %establish empty vectors to populate with x,y,z coordinates of the marble
ycoord=zeros(20,1);
zcoord=zeros(20,1);

for a=1:20
    Un(:,:,:)=reshape(Undata(a,:),n,n,n); %reshape original data
    Ut(:,:,:)=fftn(Un); %apply n-dimensional FFT to data
    Uts(:,:,:)=fftshift(Ut); %shift Ut
    Utsf(:,:,:)=filter.*Uts; %apply Gaussian filter to transformed data
    Unf(:,:,:)=ifftn(ifftshift(Utsf)); %revert filtered data to the spatial domain
    [Ms,Is]=max(Unf(:)); %find max value and its index in the filtered spatial data
    xcoord(a)=X(Is); %determine X,Y,Z max values at each of 20 time points
    ycoord(a)=Y(Is);
    zcoord(a)=Z(Is);
    plot3(xcoord,ycoord,zcoord,'.k','MarkerSize',[30],'LineWidth',2,'LineStyle','-') %plot coordinates of marble
    grid on
    xlabel('X','Fontsize',[16]); ylabel('Y','Fontsize',[16]); zlabel('Z','Fontsize',[16]);
    title('Path of marble','Fontsize',[16])    
end

%Location of marble at 20th time point
twenty=[xcoord(20),ycoord(20),zcoord(20)];
