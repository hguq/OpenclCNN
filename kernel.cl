
__kernel void conv(
    ulong CI, ulong CO, ulong H, ulong W,  // size
    __global const signed char *weight,
    __global const unsigned char* image,
    __global int *dst){
    // The input shape is [CI, H, W]
    // The weight shape is [CO, CI, 3, 3]
    // The output shape is [CO, H, W]
    int h=get_global_id(0);
    int w=get_global_id(1);
    int co=get_global_id(2);

    int acc=0;
    for(int dw=-1;dw<=1;dw++){
        for(int dh=-1;dh<=1;dh++){
            int hh=h+dh, ww=w+dw;
            int hhh=dh+1, www=dw+1;
            if(ww>=0 && ww<W && hh>=0 && hh<H){
                for(int ci=0;ci<CI;ci++){
                    acc+=weight[co*CI*3*3+ci*3*3+hhh*3+www]*image[ci*H*W+hh*W+ww];
                }
            }
        }
    }
    dst[co*H*W+h*W+w]=acc;
}

__kernel void fc(
    ulong CI, ulong CO,
    __global const signed char *weight,
    __global const unsigned char *feature,
    __global int *dst){
    // The input shape is [CI]
    // The weight shape is [CI, CO]
    // The output shape is [CO]
    int co=get_global_id(0); //only one dimension
    int acc=0;
    for(int ci=0;ci<CI;ci++){
        acc+=feature[ci]*weight[ci*CO+co];
    }
    dst[co]=acc;
}

__kernel void quan(
    ulong C, ulong H, ulong W, //if fc, H=W=1
    __global int *bias,
    __global const unsigned char *shift,
    __global const int *feature,
    __global signed char* dst){
    // The input shape is [C, H, W]
    // The bias shape is [C]
    // The shift shape is [C]
    // The output shape is [C, H, W]
    int h=get_global_id(0);
    int w=get_global_id(1);
    int c=get_global_id(2);

    int pos=c*H*W+h*W+w;
    dst[pos]=(feature[pos]-bias[c])>>shift[c];
}

__kernel void pool(
    ulong C, ulong H, ulong W, ulong HO, ulong WO,
    __global const unsigned char* feature,
    __global unsigned char* dst){
    
    int ho=get_global_id(0);
    int wo=get_global_id(1);
    int c=get_global_id(2);

    unsigned char result=0;

    for(int dh=0;dh<=1;dh++){
        for(int dw=0;dw<=1;dw++){
            int h=ho*2+dh, w=wo*2+dw;
            if(h>=0&&h<H&&w>0&&w<W)
                result=max(result, feature[c*H*W+h*W+w]);
        }
    }
    dst[c*HO*WO+ho*WO+wo]=result;
}

__kernel void relu(
    ulong C, ulong H, ulong W,
    __global signed char* feature,
    __global unsigned char* dst){
        int h=get_global_id(0);
        int w=get_global_id(1);
        int c=get_global_id(2);
        int pos=c*H*W+h*W+w;
        dst[pos]=max(0, feature[pos]);
}