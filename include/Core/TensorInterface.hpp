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

    protected:
        TensorMetadata metadata_; // Metadata of the tensor (shape, strides, offset, etc.)
        DataType dtype_;          // Data type of the tensor elements (e.g., float32, int64, etc.)
    };

}