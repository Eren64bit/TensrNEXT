#pragma once
#include "../Core/TensorMetadata.hpp"
#include "../Utils/NextTypes/NextDataType.hpp"

using TensorMetadata = NextMetadata::TensorMetadata;
using DataType = NextTypes::DataType;


namespace NextTensor
{
    /** 
     * @class TensorInterface
     * @brief An interface for tensor operations, providing methods to access metadata, data type, and raw data.
     * 
     * This interface defines the essential operations that any tensor implementation must provide.
     * It includes methods to retrieve tensor metadata, data type, and raw data pointers, as well as
     * a method to convert multi-dimensional indices to a flat index.
     * 
     * Methods:
     * - GetMetadata: Returns the metadata of the tensor (shape, strides, offset, etc.).
     * - GetDataType: Returns the data type of the tensor elements.
     * - GetRawData: Returns a mutable pointer to the raw data of the tensor.
     * - GetRawData (const): Returns a const pointer to the raw data of the tensor for read-only access.
     * - GetFlatIndex: Converts multi-dimensional indices to a flat index based on the tensor's strides.
     * 
     * Note: The actual implementation of these methods will depend on the specific tensor class that inherits from this interface.
     * **/
    class TensorInterface {
    public:
        virtual ~TensorInterface() = default;

        [[nodiscard]] virtual const TensorMetadata &GetMetadata() const noexcept = 0; // Metadata is immutable

        [[nodiscard]] virtual DataType GetDataType() const noexcept = 0; // DataType is immutable

        virtual void *GetRawData() noexcept = 0; // Raw data pointer is mutable

        [[nodiscard]] virtual const void *GetRawData() const noexcept = 0; // Const version for read-only access
    };

}