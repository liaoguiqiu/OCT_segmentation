 
x1_d= im2double (n1)*255*2;
x2_d= im2double (n2)*255*2;
x3_d= im2double (n3)*255*2;
y1 = medfilt2(x1_d,[5 5]);
y1 = medfilt2(y1,[5 5]);
y1 = imgaussfilt(y1,2); 
y2 = medfilt2(x2_d,[5 5]);
y2 = medfilt2(y2,[5 5]);
y2 = imgaussfilt(y2,2); 
y3 = medfilt2(x3_d,[5 5]);
y3 = medfilt2(y3,[5 5]);
y3 = imgaussfilt(y3,2); 
y12 =y1+y2;
y123 = y2 +y3 +y1;
y23 = y2+y3;
y1(1,1) = 1;
y2(1,1) = 1;
y3(1,1) = 1;

y123c =y123(5:195, 4:395);
y1c =y1(5:195, 4:395);
y2c =y2(5:195, 4:395);
y3c =y3(5:195, 4:395);
y23c =y23(5:195, 4:395);
y12c =y12(5:195, 4:395);