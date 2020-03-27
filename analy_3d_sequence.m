
clc;
close all;
clear all;
test_rate = 0.3;       % ????????
fileFolder=fullfile('../saved_processed_seg2/');
dirOutput=dir(fullfile(fileFolder,'*'));
fileNames={dirOutput.name}';
fullfile = char(fileNames{3:end,:});
test_num    = 100

datadir = [fileFolder,int2str(500),'.jpg'];
test_p = imread(datadir, 'jpg');
A = test_p;
Seque_num =1
for i = 500 : test_num+500
    
    datadir = [fileFolder,int2str(i),'.jpg'];
    test_p = imread(datadir, 'jpg');
%     A = cat(3,A,test_p);
    B(:,:,Seque_num)= test_p;
    Seque_num  =Seque_num+1;
end
 
[x y z] = ind2sub(size(B/255), find(B/128));
plot3(x, y, z, 'b.');
% scatter3(x,y,z, [],B/255,'filled');
figure(2)
% [X,Y,Z] = xyz2grd(B(:,1),B(:,2),B(:,3)); 
 
B (B == 0) = NaN;
S = size (B);
[X, Y, Z] = ndgrid (1: S (1), 1: S (2), 1: S (3));
scatter3 (X (:), Y (:), Z (:), 100, B (:), 'filled' )