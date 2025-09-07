#pragma once
#include <cstddef> // size_t
#include <type_traits> // is_same_v<>
#include <cstdint>

namespace NextTypes
{

    /** 
     * @enum DataType
     * @brief Enumeration of supported data types for tensors.
     * Types : FLOAT32, FLOAT64, INT32, INT64, UINT8, UINT16, UINT32, UINT64, INT8, INT16, BOOL, UNKNOWN
     * **/
    enum class DataType {
        FLOAT32,
        FLOAT64,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        INT8,
        INT16,
        BOOL,
        UNKNOWN
    };

    // Function to get the size of each data type in bytes
    /** 
     * @brief Returns the size in bytes of the specified data type.
     * @param dtype The data type whose size is to be determined.
     * @return The size in bytes of the data type. Returns 0 for unknown types
     * **/
    inline size_t GetDataTypeSize(DataType dtype) {
        switch (dtype) {
            case DataType::FLOAT32: return 4;
            case DataType::FLOAT64: return 8;
            case DataType::INT32:   return 4;
            case DataType::INT64:   return 8;
            case DataType::UINT8:   return 1;
            case DataType::UINT16:  return 2;
            case DataType::UINT32:  return 4;
            case DataType::UINT64:  return 8;
            case DataType::INT8:    return 1;
            case DataType::INT16:   return 2;
            case DataType::BOOL:    return 1; // Typically stored as a byte
            default:                return 0; // Unknown type
        }
    }

    template <typename T>
    inline DataType GetDTypeFromTemplate() {
        if (std::is_same_v<T, float>) return DataType::FLOAT32;
        if (std::is_same_v<T, double>) return DataType::FLOAT64;
        if (std::is_same_v<T, int8_t>) return DataType::INT8;
        if (std::is_same_v<T, int16_t>) return DataType::INT16;
        if (std::is_same_v<T, int>) return DataType::INT32;
        if (std::is_same_v<T, long>) return DataType::INT64;
        if (std::is_same_v<T, uint8_t>) return DataType::UINT8;
        if (std::is_same_v<T, uint16_t>) return DataType::UINT16;
        if (std::is_same_v<T, uint32_t>) return DataType::UINT32;
        if (std::is_same_v<T, uint64_t>) return DataType::UINT64;
    }
}