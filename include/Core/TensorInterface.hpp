#pragma once
#include "../Core/TensorMetadata.hpp"
#include "../Utils/NextTypes/NextDataType.hpp"

using TensorMetadata = NextMetadata::TensorMetadata;
using DataType = NextTypes::DataType;

namespace NextTensor
{
    /** 
     * @brief An interface class for tensors, providing common attributes like metadata and data type.
     * Members : metadata_, dtype_
     * **/
    class TensorInterface {
    public:
        /** 
         * @brief Constructs a TensorInterface object with the given shape, data type, and optional offset.
         * @param shape The shape of the tensor.
         * @param dtype The data type of the tensor elements.
         * @param offset The offset in the underlying data array (default is 0).
         * @return A TensorInterface object initialized with the given shape, data type, and offset.
         * **/
        TensorInterface(const TensorShapeDynamic &shape, DataType dtype, TensorOffset offset = 0) noexcept
            : metadata_(shape, offset) , dtype_(dtype) {}
            
        /** 
         * @brief Constructs a TensorInterface object with the given metadata and data type.
         * @param metadata The metadata of the tensor.
         * @param dtype The data type of the tensor elements.
         * @return A TensorInterface object initialized with the given metadata and data type.
         * **/
        TensorInterface(const TensorMetadata &metadata, DataType dtype) noexcept : metadata_(metadata), dtype_(dtype){}

        virtual ~TensorInterface() = default;
    protected:
        TensorMetadata metadata_; // Metadata of the tensor (shape, strides, offset, etc.)
        DataType dtype_;          // Data type of the tensor elements (e.g., float32, int64, etc.)
    };

}