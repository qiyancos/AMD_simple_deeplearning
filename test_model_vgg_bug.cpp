#include "test_helper.hpp"
#include "test_mpi.hpp"
#include "test_operators.hpp"

void RunSimpleVGG(HipHandle& handle){
    const int batch_size = 32;
    std::vector<int> input_shape = {batch_size, 3, 224, 224};
    std::vector<int> conv_weight_shape = {64, 3, 3, 3};
    std::vector<int> conv_bias_shape = {64, 1, 1, 1};
    std::vector<int> conv_output_shape = {batch_size, 64, 224, 224};
    std::vector<int> maxpool0_output_shape = {batch_size, 64, 112, 112};
    std::vector<int> maxpool1_output_shape = {batch_size, 64, 56, 56};
    std::vector<int> maxpool2_output_shape = {batch_size, 64, 28, 28};
    std::vector<int> maxpool3_output_shape = {batch_size, 64, 14, 14};
    std::vector<int> maxpool4_output_shape = {batch_size, 64, 7, 7};
    std::vector<int> fc_weight_shape = {1000, 64 * 7 * 7, 1, 1};
    std::vector<int> fc_bias_shape = {1000, 1, 1, 1};
    std::vector<int> output_shape = {batch_size, 1000, 1, 1};
    std::vector<int> loss_shape = {1, 1, 1, 1};

    ConvDescriptor convSpec("conv", 1, 1, 1, 1);
    PoolingDescriptor poolSpec("max", 2, 2, 0, 0, 2, 2);
    
    Tensor<float> input(1, input_shape);
    Tensor<float> conv_weight(1, conv_weight_shape);
    Tensor<float> conv_bias(1, conv_bias_shape);
    Tensor<float> conv_output(conv_output_shape);
    Tensor<float> maxpool0_output(maxpool0_output_shape);
    Tensor<float> maxpool1_output(maxpool1_output_shape);
    Tensor<float> maxpool2_output(maxpool2_output_shape);
    Tensor<float> maxpool3_output(maxpool3_output_shape);
    Tensor<float> maxpool4_output(maxpool4_output_shape);
    Tensor<float> fc_weight(1, fc_weight_shape);
    Tensor<float> fc_bias(1, fc_bias_shape);
    Tensor<float> output(output_shape);
    Tensor<float> loss(loss_shape);

    Tensor<float> input_grad(input_shape);
    Tensor<float> conv_weight_grad(conv_weight_shape);
    Tensor<float> conv_bias_grad(conv_bias_shape);
    Tensor<float> conv_output_grad(conv_output_shape);
    Tensor<float> maxpool0_output_grad(maxpool0_output_shape);
    Tensor<float> maxpool1_output_grad(maxpool1_output_shape);
    Tensor<float> maxpool2_output_grad(maxpool2_output_shape);
    Tensor<float> maxpool3_output_grad(maxpool3_output_shape);
    Tensor<float> maxpool4_output_grad(maxpool4_output_shape);
    Tensor<float> fc_weight_grad(fc_weight_shape);
    Tensor<float> fc_bias_grad(fc_bias_shape);
    Tensor<float> output_grad(1, output_shape);
    Tensor<float> loss_grad(loss_shape);

    ConvolutionOp<float>::ConvForward(handle, convSpec,
            input, conv_weight, &conv_bias, conv_output);
    PoolingOp<float>::PoolingForward(handle, poolSpec,
            conv_output, maxpool0_output);
    PoolingOp<float>::PoolingForward(handle, poolSpec,
            maxpool0_output, maxpool1_output);
    PoolingOp<float>::PoolingForward(handle, poolSpec,
            maxpool1_output, maxpool2_output);
    PoolingOp<float>::PoolingForward(handle, poolSpec,
            maxpool2_output, maxpool3_output);
    PoolingOp<float>::PoolingForward(handle, poolSpec,
            maxpool3_output, maxpool4_output);
    FullyConnectOp<float>::FullyConnectForward(handle,
            maxpool4_output, fc_weight, &fc_bias, output);
    // CrossEntropyOp<float>::CrossEntropyForward();

    // CrossEntropyOp<float>::CrossEntropyBackward();
    FullyConnectOp<float>::FullyConnectBackwardData(handle,
            output_grad, fc_weight, maxpool4_output_grad);
    FullyConnectOp<float>::FullyConnectBackwardWeight(handle,
            output_grad, maxpool4_output, fc_weight_grad, &fc_bias_grad);
    PoolingOp<float>::PoolingBackward(handle, poolSpec,
            maxpool3_output, maxpool4_output,
            maxpool4_output_grad, maxpool3_output_grad);
    PoolingOp<float>::PoolingBackward(handle, poolSpec,
            maxpool2_output, maxpool3_output,
            maxpool3_output_grad, maxpool2_output_grad);
    PoolingOp<float>::PoolingBackward(handle, poolSpec,
            maxpool1_output, maxpool2_output,
            maxpool2_output_grad, maxpool1_output_grad);
    PoolingOp<float>::PoolingBackward(handle, poolSpec,
            maxpool0_output, maxpool1_output,
            maxpool1_output_grad, maxpool0_output_grad);
    PoolingOp<float>::PoolingBackward(handle, poolSpec,
            conv_output, maxpool0_output,
            maxpool0_output_grad, conv_output_grad);
    ConvolutionOp<float>::ConvBackwardWeight(handle, convSpec,
            conv_output_grad, input,
            conv_weight_grad, &conv_bias_grad);
    ConvolutionOp<float>::ConvBackwardData(handle, convSpec,
            conv_output_grad, conv_weight, input_grad);
}

int main(int argc, char** argv){
    Communicator comm(argc, argv);
    HipHandle handle(comm.getRank());
    CHECK_CALL_HIP(hipSetDevice(comm.getRank()));

    int testIters = 500;
    for(int i = 0; i < testIters; i++){
        std::cout << "Pid-" << getpid() << ": Running Iter " << i << std::endl;
        RunSimpleVGG(handle);
    }

    return 0;
}
