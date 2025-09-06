#pragma once

namespace NextTypes
{
    /** *
     * @enum Device Type
     * @brief Enumeration of supported device types for tensor computations.
     * Types : CPU, GPU, TPU, UNKNOWN
    */
    enum class DeviceType {
        CPU,
        GPU,
        TPU,
        UNKNOWN
    };
}