%enter the labels alone, if already shaped
%with heigth and width to reform the image
%or just with the picture, to blend image and labels
function map = display_map(labels,height_or_picture,width)

    if nargin==1
        [height,width]=size(labels);
    end
    if nargin==3
        labels=reshape(labels,height_or_picture,width);
        height=height_or_picture;
    end
    if nargin==2
        [height,width,dim]=size(height_or_picture);
        picture=height_or_picture;
    end
    
    r=zeros(height,width,1);
    g=zeros(height,width,1);
    b=zeros(height,width,1);

    i=(labels==1);
    b(i)=1.0;

    i=(labels==2);
    g(i)=1.0;

    i=(labels==3);
    r(i)=1.0;

    i=(labels==4);
    r(i)=1.0;
    g(i)=1.0;

    i=(labels==5);
    r(i)=1.0;
    b(i)=1.0;

    i=(labels==6);
    g(i)=1.0;
    b(i)=1.0;
    
    i=(labels==7);
    r(i)=1.0;
    g(i)=0.55;

    
    i=(labels==8);
    r(i)=0.5;
    g(i)=0.1;
    b(i)=0.55;
    
    i=(labels==9);
    r(i)=0.2;
    g(i)=0.34;
    b(i)=0.34;
    
    map=cat(3,r,g,b);
    
    if nargin==2
        map=(double(map)+3*double(picture)/255.0)/4;
    end
    
    imagesc(map);


end