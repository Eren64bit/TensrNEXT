#pragma once

#include "../Utils/NextUtils.hpp"


namespace NextMetadata
{
    /** 
     * @brief A class to hold metadata about a tensor, including its shape, strides, offset, total size, and contiguity.
     * Functions : Getters, Setters, Print
     * **/
    class TensorMetadata {
    protected:
        TensorShapeDynamic shape_;    // Shape of the tensor (e.g., {2, 3, 4} for a 2x3x4 tensor) TensorShapeDynamic = std::vector<size_t>
        TensorStrideDynamic strides_; // Strides of the tensor (e.g., {12, 4, 1} for a 2x3x4 tensor) TensorStrideDynamic = std::vector<size_t>
        TensorOffset offset_;         // Offset in the underlying data array
        TensorSize totalSize_;        // Total number of elements in the tensor
        TensorRank rank_;             // Rank (number of dimensions) of the tensor
        bool isContiguous_;           // Whether the tensor is stored in contiguous memory
    public:
    
        // Constructor
        /** 
         * @brief Constructs a TensorMetadata object with the given shape and optional offset.
         * @param shape The shape of the tensor.
         * @param offset The offset in the underlying data array (default is 0).
         * @return A TensorMetadata object initialized with the given shape and offset.
         * **/
        TensorMetadata(const TensorShapeDynamic &shape, TensorOffset offset = 0) noexcept
            : shape_(shape), offset_(offset){
                strides_ = NextUtils::ComputeStrides(shape_);
                totalSize_ = NextUtils::ComputeSize(shape_);
                rank_ = shape_.size();
                isContiguous_ = true; // Assuming newly created tensors are contiguous
        }

        /** 
         * @brief Constructs a TensorMetadata object with the given shape, strides, and optional offset.
         * @param shape The shape of the tensor.
         * @param stride The strides of the tensor.
         * @param offset The offset in the underlying data array (default is 0).
         * @return A TensorMetadata object initialized with the given shape, strides, and offset.
         * **/
        TensorMetadata(const TensorShapeDynamic &shape, TensorStrideDynamic &stride, TensorOffset offset = 0) noexcept
            : shape_(shape), strides_(stride), offset_(offset){
                totalSize_ = NextUtils::ComputeSize(shape_);
                rank_ = shape_.size();
                isContiguous_ = true; // Assuming newly created tensors are contiguous
        }

        /** 
         * @brief Gets the shape of the tensor.
         * @param shape The new shape of the tensor.
         * @return A reference to the shape of the Metadata object.
         * **/
        [[nodiscard]] const TensorShapeDynamic& GetShape() const noexcept { return shape_; }

        /** 
         * @brief Gets the strides of the tensor.
         * @param shape The new shape of the tensor.
         * @return A reference to the strides of the Metada object.
         * **/  
        [[nodiscard]] const TensorStrideDynamic& GetStrides() const noexcept { return strides_; }

        /** 
         * @brief Gets the offset of the tensor.
         * @return The offset of the tensor.
         * **/
        [[nodiscard]] TensorOffset GetOffset() const noexcept { return offset_; }

        /** 
         * @brief Gets the total size of the tensor.
         * @return The total number of elements in the tensor.
         * **/
        [[nodiscard]] TensorSize GetTotalSize() const noexcept { return totalSize_; }

        /** 
         * @brief Gets the rank of the tensor.
         * @return The rank (number of dimensions) of the tensor.
         * **/
        [[nodiscard]] TensorRank GetRank() const noexcept { return rank_; }

        /** 
         * @brief Checks if the tensor is contiguous in memory.
         * @return True if the tensor is contiguous, false otherwise.
         * **/
        [[nodiscard]] bool IsContiguous() const noexcept { return isContiguous_; }

        /** 
         * @brief Sets the contiguity of the tensor.
         * @param contiguous True if the tensor is contiguous, false otherwise.
         * **/
        void SetContiguous(bool contiguous) noexcept { isContiguous_ = contiguous; }

    };

}