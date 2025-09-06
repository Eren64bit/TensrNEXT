#pragma once

namespace NextTypes
{
    /** 
     * @enum Memory Layout
     * @brief Enumeration for different memory layouts of tensors.
     * Values : ROW_MAJOR, COLUMN_MAJOR, UNKNOWN
     * **/
    enum class MemoryLayout {
        ROW_MAJOR,
        COLUMN_MAJOR,
        UNKNOWN
    };
}