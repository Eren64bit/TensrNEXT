#pragma once 

#include "TensorInterface.hpp"
#include <memory> // std::unique_ptr
#include <array>  // std::array

namespace NextTensor
{

    /** 
     * @class TensorStatic
     * @brief A class representing a tensor with static shape and size, inheriting from TensorInterface.
     * 
     * This class is designed to handle tensors with a fixed shape and size at compile time, allowing for
     * optimizations in memory layout and access patterns. It provides methods for accessing and manipulating
     * the tensor data, as well as retrieving metadata about the tensor.
     * Handle data storage using std::array for fixed size arrays.
     */
    template <typename T, TensorSize N> // TensorSize = size_t
    class TensorStatic : public TensorInterface {
    public:
        ~TensorStatic() = default;
        TensorStatic(const TensorStatic&) = delete;
        TensorStatic& operator=(const TensorStatic&) = delete;
        // Constructor

        /** 
         * @brief TensorStatic Constructor with Shape param
         * @param shape: Shape of the tensor
         * **/
        TensorStatic(const TensorShapeStatic<N> &shape)
            : metadata_(shape),
              dtype_(NextTypes::GetDTypeFromTemplate<T>()),
              data_(std::make_unique<std::array<T, N>>()) {}

        /** 
         * @brief TensorStatic Constructor with Metadata param
         * @param metadata: Metadata of the tensor
         * **/
        TensorStatic(const TensorMetadata &metadata)
            : metadata_(metadata),
              dtype_(NextTypes::GetDTypeFromTemplate<T>()),
              data_(std::make_unique<std::array<T, N>>()) {}

        // Interface implementations
        /** 
         * @brief Returns the metadata of the tensor (shape, strides, offset, etc.).
         * Note: Implementation of TensorInterface virtual function
         * @return A constant reference to the TensorMetadata object.
         * **/
        [[nodiscard]] const TensorMetadata &GetMetadata() const noexcept override { return metadata_; }

        /** 
         * @brief Returns the Data Type of the Tensor elements.
         * Note: Implementation of TensorInterface virtual function
         * @return The DataType of the tensor elements.
         * **/
        [[nodiscard]] DataType GetDataType() const noexcept override { return dtype_; }

        /** 
         * @brief Returns the Raw pointer of the tensor data array.
         * Note: Implementation of TensorInterface virtual function
         * @return The Raw pointer to the Tensor data.
         * **/
        void *GetRawData() noexcept override { return data_.get(); }

        /** 
         * @brief Returns The Raw pointer of the Tensor data array.
         * Note: Implementation of TensorInterface virtual function
         * @return The Raw pointer to the Tensor data.
         * Note: Const version read-only access.
         * **/
        [[nodiscard]] const void *GetRawData() const noexcept override { return data_.get(); }

    private:
        TensorMetadata metadata_; // Metadata of the tensor (shape, strides, offset, etc.)
        DataType dtype_;          // Data type of the tensor elements
        std::unique_ptr<std::array<T, N>> data_; // Unique pointer to the tensor data array        
    };
}