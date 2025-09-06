#pragma once

namespace NextTypes
{
    /** 
     * @enum Operation Type
     * @brief Enumeration for different types of tensor operations.
     * Values : ADD, SUB, MUL, DIV, MATMUL, RELU, SIGMOID, TANH, SOFTMAX, CONV2D, MAXPOOL, AVGPOOL, FLATTEN, RESHAPE, TRANSPOSE, UNKNOWN
     * **/
    enum class OpType {
        ADD,
        SUB,
        MUL,
        DIV,
        MATMUL,
        RELU,
        SIGMOID,
        TANH,
        SOFTMAX,
        CONV2D,
        MAXPOOL,
        AVGPOOL,
        FLATTEN,
        RESHAPE,
        TRANSPOSE,
        UNKNOWN
    };
}