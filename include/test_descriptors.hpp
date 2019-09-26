#ifndef TEST_DESCRIPTORS_HPP
#define TEST_DESCRIPTORS_HPP

// Convolution
struct ConvDescriptor {
    std::string mode;
    int convdim = 2;
    int group = 1;
    std::vector<int> padding;
    std::vector<int> stride;
    std::vector<int> dilation;
    ConvDescriptor(const std::string mode_in,
            const int padding_h, const int padding_w,
            const int stride_h, const int stride_w) {
        mode = mode_in;
        padding.push_back(padding_h);
        padding.push_back(padding_w);
        stride.push_back(stride_h);
        stride.push_back(stride_w);
    }
    ConvDescriptor(const std::string mode_in,
            const int padding_h, int padding_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w) {
        mode = mode_in;
        padding.push_back(padding_h);
        padding.push_back(padding_w);
        stride.push_back(stride_h);
        stride.push_back(stride_w);
        dilation.push_back(dilation_h);
        dilation.push_back(dilation_w);
    }
    miopenConvolutionMode_t getMode(){
        if(mode == "conv") {
            return miopenConvolution;
        } else if(mode == "deconv") {
            return miopenTranspose;
        } else {
            std::cerr << "Error: Unknown convolution mode!" << std::endl;
            exit(1);
        }
    }
};

// Pooling
struct PoolingDescriptor {
    int pooldim = 2;
    std::string mode;
    std::vector<int> kernelshape;
    std::vector<int> padding;
    std::vector<int> stride;
    PoolingDescriptor(std::string mode_in,
            int kernel_h, int kernel_w, 
            int padding_h, int padding_w,
            int stride_h, int stride_w){
        mode = mode_in;
        kernelshape.push_back(kernel_h);
        kernelshape.push_back(kernel_w);
        padding.push_back(padding_h);
        padding.push_back(padding_w);
        stride.push_back(stride_h);
        stride.push_back(stride_w);
    }
    miopenPoolingMode_t getMode(){
        if(mode == "avg") {
            return miopenPoolingAverage;
        } else if(mode == "max") {
            return miopenPoolingMax;
        } else {
            std::cerr << "Error: Unknown pooling mode!" << std::endl;
            exit(1);
        }
    }
};

#endif
