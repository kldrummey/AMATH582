%% AMATH582 - Homework 2

%% Handel's Messiah 
clear all; close all; clc;

%Load in data
load handel; H=y'/2;
haudio=audioplayer(H,Fs); playblocking(haudio);

figure(1)
plot((1:length(H))/Fs,H,'k'); xlabel('Time (s)')
ylabel('Amplitude'); title('Raw signal of interest, H(n)')

%Establish parameters and frequency range
n=length(H); L=n/Fs; t=(1:n)/Fs;
k=[0:(Fs/2)/(n/2-1):Fs/2]; 

%Fourier transform and plot of unfiltered data
Ht=fft(H);  

figure(2)
plot(k,Ht(1:length(Hts)/2),'k'); xlabel('Frequency (Hz)','FontSize',[14])
ylabel('Amplitude','FontSize',[14]); title('Fourier Transform of H(n)','FontSize',[14])
xlim([0 2000]);

%Gabor windowing and plot - filter widths of 1, 10, 50, and 100
g1=exp(-1.*(t-4).^2); Hg1=g1.*H; Hg1t=fft(Hg1);

g2=exp(-10.*(t-4).^2); Hg2=g2.*H; Hg2t=fft(Hg2);

g3=exp(-50.*(t-4).^2); Hg3=g3.*H; Hg3t=fft(Hg3);

g4=exp(-100.*(t-4).^2);Hg4=g4.*H; Hg4t=fft(Hg4); 

figure(3)
subplot(4,1,1)
plot(k,abs(Hg1t(1:length(Hg1t)/2)),'k'); title('Gabor window width of 1'); ylabel('Amplitude')
subplot(4,1,2)
plot(k,abs(Hg2t(1:length(Hg2t)/2)),'k'); title('Gabor window width of 10'); ylabel('Amplitude')
subplot(4,1,3)
plot(k,abs(Hg3t(1:length(Hg3t)/2)),'k'); title('Gabor window width of 50'); ylabel('Amplitude')
subplot(4,1,4)
plot(k,abs(Hg4t(1:length(Hg4t)/2)),'k'); xlabel('Frequencies'); title('Gabor window width of 100'); ylabel('Amplitude')
xlim([0 4000])

%% Handel spectrograms using different window widths

%Spectrogram with width 1
tslide=0:0.1:L; 
Hg1_spec=zeros(length(tslide),length(H)); 
for jj=1:length(tslide)
    g1=exp((-1*(t-tslide(jj)).^2)).*cos((t-tslide(jj))*pi);
    Hg1=g1.*H;
    Hg1t=fft(Hg1);
    Hg1_spec(jj,:)=abs(Hg1t);
end
Hg1_spec=Hg1_spec(1:length(tslide),1:length(Hg1_spec)/2);

f=figure('Visible',false);
pcolor(tslide,k,Hg1_spec.'); ylim([0 2000])
shading interp; colormap(hot); colorbar;
print('Handel_Width1.jpeg','-djpeg')
close(f)

%Spectrogram with width 10
tslide=0:0.1:L; 
Hg10_spec=zeros(length(tslide),length(H)); 
for jj=1:length(tslide)
    g10=exp((-10*(t-tslide(jj)).^2)).*cos((t-tslide(jj))*pi);
    Hg10=g10.*H;
    Hg10t=fft(Hg10);
    Hg10_spec(jj,:)=abs(Hg10t);
end
Hg10_spec=Hg10_spec(1:length(tslide),1:length(Hg10_spec)/2);

f=figure('Visible',false);
pcolor(tslide,k,Hg10_spec.'); ylim([0 2000])
shading interp; colormap(hot); colorbar;
print('Handel_Width10.jpeg','-djpeg')
close(f)

%Spectrogram with width 30
tslide=0:0.1:L; 
Hg30_spec=zeros(length(tslide),length(H)); 
for jj=1:length(tslide)
    g30=exp((-30*(t-tslide(jj)).^2)).*cos((t-tslide(jj))*pi);
    Hg30=g30.*H;
    Hg30t=fft(Hg30);
    Hg30_spec(jj,:)=abs(Hg30t);
end
Hg30_spec=Hg30_spec(1:length(tslide),1:length(Hg30_spec)/2);

f=figure('Visible',false);
pcolor(tslide,k,Hg30_spec.'); ylim([0 2000])
shading interp; colormap(hot); colorbar;
print('Handel_Width30.jpeg','-djpeg')
close(f)

%% Handel spectrograms with varying timesteps, window width of 10
tslide5=0:0.05:L; %Timestep of 0.05
Hg5_spec=zeros(length(tslide5),length(H)); 
for jj=1:length(tslide5)
    g5=exp((-10*(t-tslide5(jj)).^2)).*cos((t-tslide5(jj))*pi);
    Hg5=g5.*H;
    Hg5t=fft(Hg5);
    Hg5_spec(jj,:)=abs(Hg5t);
end
Hg5_spec=Hg5_spec(1:length(tslide5),1:length(Hg5_spec)/2);

f=figure('Visible',false);
pcolor(tslide5,k,Hg5_spec.'); ylim([0 2000])
shading interp; colormap(hot); colorbar;
print('Handel_TimeStep0.05.jpeg','-djpeg')
close(f)

%Timestep of 1
tslide1=0:1:L; 
Hg1_spec=zeros(length(tslide1),length(H)); %Spectrogram with width 10
for jj=1:length(tslide1)
    g1=exp((-10*(t-tslide1(jj)).^2)).*cos((t-tslide1(jj))*pi);
    Hg1=g1.*H;
    Hg1t=fft(Hg1);
    Hg1_spec(jj,:)=abs(Hg1t);
end
Hg1_spec=Hg1_spec(1:length(tslide1),1:length(Hg1_spec)/2);

f=figure('Visible',false);
pcolor(tslide1,k,Hg1_spec.'); ylim([0 2000])
shading interp; colormap(hot); colorbar;
print('Handel_TimeStep1.jpeg','-djpeg')
close(f)

%Timestep of 2
tslide2=0:2:L; 
Hg2_spec=zeros(length(tslide2),length(H)); %Spectrogram with width 10
for jj=1:length(tslide2)
    g2=exp((-10*(t-tslide2(jj)).^2)).*cos((t-tslide2(jj))*pi);
    Hg2=g2.*H;
    Hg2t=fft(Hg2);
    Hg2_spec(jj,:)=abs(Hg2t);
end
Hg2_spec=Hg2_spec(1:length(tslide2),1:length(Hg2_spec)/2);

f=figure('Visible',false);
pcolor(tslide2,k,Hg2_spec.'); ylim([0 2000])
shading interp; colormap(hot); colorbar;
print('Handel_TimeStep2.jpeg','-djpeg')
close(f)

%% Mary had a little lamb 
%% Setting up parameters and determining frequency components
close all; clear all; clc; 

tp=16;  % record time in seconds - piano
p=audioread('music1.wav','native'); 
p=double(p); np=length(p'); Fsp=np/tp; Lp=np/Fsp; tp=(1:np)/Fsp;
kp=[0:(Fsp/2)/(np/2-1):Fsp/2];
plot((1:np)/Fsp,p); xlabel('Time [sec]'); ylabel('Amplitude'); title('Mary had a little lamb (piano)');  
% p8 = audioplayer(p,Fs); playblocking(p8);

tr=14;  % record time in seconds
r=audioread('music2.wav','native'); 
r=double(r); nr=length(r'); Fsr=nr/tr; Lr=nr/Fsr; tr=(1:nr)/Fsr;
kr=[0:(Fsr/2)/(nr/2-1):Fsr/2];
plot((1:nr)/Fsr,r); xlabel('Time [sec]'); ylabel('Amplitude'); title('Mary had a little lamb (recorder)');
% p8 = audioplayer(y,Fs); playblocking(p8);

%Look at frequenices included in the songs
pt=fft(p); rt=fft(r);

figure(1)
hold on
plot(kp,abs(pt(1:length(pt)/2)),'k')
plot(kr,abs(rt(1:length(rt)/2)),'b')
xlim([0 4000]);title('Frequency components, piano vs. recorder','Fontsize',[14])
ylabel('Amplitude','Fontsize',[16]); xlabel('Frequency, Hz','Fontsize',[16])
xticks([0:200:4000]); legend('Piano','Recorder','Fontsize',[20]);


%% Spectrograms for recorder and piano pieces
middleCoctave=[261.63,293.66,329.63,349.23,392,440,493.88,523.55]; %Scale from middle C to 1 octave above

tpslide=0:0.05:Lp;
pspec=zeros(length(tpslide),length(p));
for jj=1:length(tpslide)
    g_piano=exp((-20*(tp-tpslide(jj)).^2)).*cos((tp-tpslide(jj))*pi);
    pg=g_piano.*p';
    pgt=fft(pg);
    pspec(jj,:)=abs(pgt);
end
pspec=pspec(1:length(tpslide),1:length(pspec)/2);

octaveP=zeros(length(tpslide),length(middleCoctave));
for o=1:length(octaveP)
    octaveP(o,:)=middleCoctave(:);
end

f=figure('Visible',false);
hold on
pcolor(tpslide,kp,pspec.')
shading interp; colormap(hot); colorbar
plot(octaveP,'-w')
xlim([0 max(tpslide)]); ylim([0 1000]);
xlabel('Time (s)'); ylabel('Frequency (Hz)'); title('Spectrogram - Piano')
print('PianoSpec.jpeg','-djpeg')
close(f)

highCoctave=[523.55, 587.33,659.26,698.46,783.99,880,987.77,1049.5,1174.7];
trslide=0:0.05:Lr;
rspec=zeros(length(trslide),length(r));
for j=1:length(trslide)
    g_rec=exp((-20*(tr-trslide(j)).^2)).*cos((tr-trslide(j))*pi);
    rg=g_rec.*r';
    rgt=fft(rg);
    rspec(j,:)=abs(rgt);
end
rspec=rspec(1:length(trslide), 1:length(rspec)/2);

octaveR=zeros(length(trslide),length(highCoctave));
for o=1:length(octaveR)
    octaveR(o,:)=highCoctave(:);
end

f=figure('Visible',false);
hold on
pcolor(trslide,kr,rspec.'); 
shading interp; colormap(hot); colorbar; ylim([0 1500])
plot(octaveR,'-w')
xlim([0 max(trslide)]);
xlabel('Time (s)'); ylabel('Frequency (Hz)'); title('Spectrogram - Recorder')
print('RecorderSpec.jpeg','-djpeg')
close(f)


%% Determine overtones in piano and recorder signals
%Look at number of instances where spectrogram data is within 1, 2, or 3
%harmonics of first note in song

%~330Hz=E, first note in piano score; ~1050Hz=C, first note in recorder
%score
countph1=0; countph2=0; countph3=0;
for j=1:length(pspec)
    if 330<pspec(j) && pspec(j)<(330*2)
        countph1=countph1+1;
    elseif (330*2)<pspec(j) && pspec(j)<(330*3)
        countph2=countph2+1;
    elseif (330*3)<pspec(j) && pspec(j)<(330*4)
        countph3=countph3+1;
    end
end

countrh1=0; countrh2=0; countrh3=0;
for j=1:length(rspec)
    if 1050<rspec(j) && rspec(j)<(1050*2)
        countrh1=countrh1+1;
    elseif (1050*2)<rspec(j) && rspec(j)<(1050*3)
        countrh2=countrh2+1;
    elseif (1050*3)<rspec(j) && rspec(j)<(1050*4)
        countrh3=countrh3+1;
    end
end

countsp=[countph1 countph2 countph3]; countsr=[countrh1 countrh2 countrh3]; harmonics=[1 2 3];

figure()
hold on
plot(harmonics, countsp,'ok','MarkerSize',[10],'MarkerFaceColor','k')
plot(harmonics, countsr,'or','MarkerSize',[10],'MarkerFaceColor','r')
xlabel('Harmonic','FontSize',[14]); ylabel('Number of instances','FontSize',[14]); 
title('Instances of harmonic values for first note in recorder vs piano','FontSize',[14])
legend('Piano','Recorder','FontSize',[16]); xticks([0:1:3]);

