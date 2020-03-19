%% AMATH582 Final Project - Analysis of ECoG Signals
clear all; close all; clc; 
%% Clean up inputs and assign file names etc.
%Input file name for analysis; enter filename surrounded by quotes
fileName=input('Enter name of file for analysis')

%Call load_file_emg.m to convert hex files to integers
load_file_emg_edited

%Call in structure of channels recorded on day/time
dataFiles=csv2struct('datafiles_channels.xlsx');

%Rename fileName to match file name input into struct
fileName=regexprep(fileName,'.txt','_txt');

%Pull out information about which channels were recorded from and date/time. 
fldnm=fileName;

if isfield(dataFiles,fileName)==1
    expInfo=extractfield(dataFiles,fldnm)
end

info=expInfo{1,1};
channel1Name=info{3,1};
channel2Name=info{4,1};
experimentDate=info{1,1};
experimentStartTime=info{2,1};

%% Raw data graphs
%convert sample rate to time 
recording_length=(length(ch1_s))/fs;
time=linspace(1,recording_length,length(ch1_s));

figure(1)
subplot(2,1,1)
plot(time,ch1_s,'k')
ylabel('mV')
title(sprintf(channel1Name))
subplot(2,1,2)
plot(time,ch2_s,'k')
ylabel('mV')
title(sprintf(channel2Name))
xlabel('Time (s)')

%% FFT and power spectral density of data 
n1=length(ch1_s); n2=length(ch2_s);
L1=n1/fs; L2=n2/fs; t1=(1:n1)/fs; t2=(1:n2)/fs;
k1=[0:(fs/2)/(n1/2-1):fs/2]; k2=[0:(fs/2)/(n2/2-1):fs/2]; 
ch1_bp=bandpass(ch1_s,[5 250],fs); ch2_bp=bandpass(ch2_s,[5 250], fs);

%FFT
ch1_bpt=abs(fft(ch1_bp)); ch2_bpt=abs(fft(ch2_bp));

figure(1)
subplot(2,1,1); plot(k1,ch1_bpt(1:length(ch1_bpt)/2),'k'); xlim([5 250]); 
ylabel('Amplitude'); xlabel('Frequency (Hz)'); title(sprintf(channel1Name));
subplot(2,1,2); plot(k2,ch2_bpt(1:length(ch2_bpt)/2),'k'); xlim([5 250]); 
ylabel('Amplitude'); xlabel('Frequency (Hz)'); title(sprintf(channel2Name));
sgtitle('Frequency components, channels 1 and 2','FontSize',[14])

%PSD
[Pxx1,f1]=pwelch((ch1_bp)-mean(ch2_bp),[],[],fs);
[Pxx2,f2]=pwelch((ch2_bp)-mean(ch2_bp),[],[],fs);

figure(2)
subplot(2,1,1);plot(Pxx1,'k','LineWidth',1.25); xlim([1 250]); title('Channel 1');
xlabel('Frequency (Hz)'); ylabel('PSD Estimate (mV^2/Hz)');
subplot(2,1,2);plot(Pxx2,'k','LineWidth',1.25); xlim([1 250]); title('Channel 2');
xlabel('Frequency (Hz)'); ylabel('PSD Estimate (mV^2/Hz)');
sgtitle('Power Spectral Density Plots, Channels 1 and 2')
%% Gabor filtering and spectrogram
ch1_cut=ch1_bp(1:2500); ch2_cut=ch2_bp(1:2500); %Take segment of data so memory doesn't overload

%Set parameters
n1=length(ch1_cut); n2=length(ch2_cut);
L1=n1/fs; L2=n2/fs; t1=(1:n1)/fs; t2=(1:n2)/fs;
k1=[0:(fs/2)/(n1/2-1):fs/2]; k2=[0:(fs/2)/(n2/2-1):fs/2]; 

%Gabor filter and spectrogram
tslide=0:0.1:L1;
ch1_spec=zeros(length(tslide),length(ch1_cut));

for j=1:length(tslide)
    g=exp((-25*(t1-tslide(j)).^2)).*cos((t1-tslide(j))*pi);
    ch1g=g.*ch1_cut;
    ch1gt=fft(ch1g);
    ch1_spec=[ch1_spec; abs(ch1gt)];
end
ch1_spec=ch1_spec(1:length(tslide),1:length(ch1_cut)/2);

f=figure('Visible',false);
pcolor(tslide,k1,ch1_spec.'); ylim([0 250])
shading interp; colorbar;
print('DIFF0_Width25_11052019.jpeg','-djpeg')
close(f)


