#pragma once

#include <cstddef> // size_t
#include <vector>  // std::vector
#include <array>   // std::array
#include <stdexcept> // std::invalid_argument, std::out_of_range


template <size_t N>
using TensorStrideStatic = std::array<size_t, N>; // Stride of a tensor with static rank
template <size_t N>
using TensorShapeStatic = std::array<size_t, N>;  // Shape of a tensor with static rank

using TensorShapeDynamic = std::vector<size_t>;   // Shape of a tensor with dynamic rank
using TensorStrideDynamic = std::vector<size_t>;  // Stride of a tensor with dynamic rank

using TensorSize  = size_t;                       // Size of a tensor dimension
using TensorIndex = size_t;                       // Index of a tensor element
using TensorOffset = size_t;                      // Offset in the tensor's underlying data array
using TensorRank = size_t;                        // Rank (number of dimensions) of a tensor
template <size_t N>
using TensorIndexStatic = std::array<size_t, N>;  // Shape of a tensor with static rank
using TensorIndexDynamic = std::vector<size_t>;   // Shape of a tensor with dynamic rank

/** 
 * @namespace NextUtils
 * @brief A namespace for utility functions and definitions used in the project.
 * Functions : ComputeStrides, ComputeSize, FlattenIndex, UnflattenIndex
 * 
 * **/
namespace NextUtils
{
    /** 
     * @brief Computes the strides for a tensor given its shape.
     * @tparam N The rank (number of dimensions) of the tensor.
     * @param shape The shape of the tensor.
     * @return The computed strides for the tensor.
     * ConstExpr version for static rank tensors.
     * **/
    template <size_t N>
    [[nodiscard]] constexpr TensorStrideStatic<N> ComputeStrides(const TensorShapeStatic<N>& shape) noexcept{
        TensorStrideStatic<N> strides{};

        if constexpr (N > 0) {
            strides[N- 1] = 1; // Last dimension stride is always 1
            for (int i = static_cast<int>(N - 2); i >= 0; --i) {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        return strides;
    }

    /** 
     * @brief Computes the strides for a tensor given its shape.
     * @param shape The shape of the tensor.
     * @return The computed strides for the tensor.
     * Runtime version for dynamic rank tensors.
     * **/
    [[nodiscard]] inline TensorStrideDynamic ComputeStrides(const TensorShapeDynamic& shape) noexcept {
        size_t N = shape.size();
        TensorStrideDynamic strides(N, 1);

        for (int i = static_cast<int>(N) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    /** 
     * @brief Computes the total size (number of elements) of a tensor given its shape.
     * @tparam N The rank (number of dimensions) of the tensor.
     * @param shape The shape of the tensor.
     * @return The total size of the tensor.
     * ConstExpr version for static rank tensors.
     * **/
    template <size_t N>
    [[nodiscard]] constexpr TensorSize ComputeSize(const TensorShapeStatic<N>& shape) noexcept {
        TensorSize size = 1;
        if constexpr (N > 0) {
            for (const auto& dim : shape) {
                size *= dim;
            }
        }
        return size;
    }

    /** 
     * @brief Computes the total size (number of elements) of a tensor given its shape.
     * @param shape The shape of the tensor.
     * @return The total size of the tensor.
     * Runtime version for dynamic rank tensors.
     * **/
    [[nodiscard]] inline TensorSize ComputeSize(const TensorShapeDynamic& shape) noexcept {
        TensorSize size = 1;
        for (const auto& dim : shape) {
            size *= dim;
        }
        return size;
    }

    /** 
     * @brief Flattens multi-dimensional indices into a single-dimensional index using the provided strides.
     * @tparam N The rank (number of dimensions) of the tensor.
     * @param strides The strides of the tensor.
     * @param indices The multi-dimensional indices to flatten.
     * @return The flattened single-dimensional index.
     * ConstExpr version for static rank tensors.
     * **/
    template <size_t N>
    [[nodiscard]] constexpr TensorIndex FlattenIndex(const TensorStrideStatic<N> &strides, const TensorIndexStatic<N> &indices) noexcept {
        if constexpr (N == 0) {
            return 0; // If there are no dimensions, the flat index is always 0
        }
        TensorIndex FlatIndex = 0;
        for (size_t i = 0; i < N; ++i) {
            FlatIndex += indices[i] * strides[i];
        }
        return FlatIndex;
    }

    /** 
     * @brief Flattens multi-dimensional indices into a single-dimensional index using the provided strides.
     * @param strides The strides of the tensor.
     * @param indices The multi-dimensional indices to flatten.
     * @return The flattened single-dimensional index.
     * Runtime version for dynamic rank tensors.
     * **/
    [[nodiscard]] inline TensorIndex FlattenIndex(const TensorStrideDynamic &strides, const TensorIndexDynamic &indices) noexcept {
        size_t N = strides.size();
        TensorIndex FlatIndex = 0;
        for (size_t i = 0; i < N; ++i) {
            FlatIndex += indices[i] * strides[i];
        }
        return FlatIndex;
    }

    /** 
     * @brief Unflattens a single-dimensional index into multi-dimensional indices using the provided strides.
     * @tparam N The rank (number of dimensions) of the tensor.
     * @param strides The strides of the tensor.
     * @param flatIndex The single-dimensional index to unflatten.
     * @return The unflattened multi-dimensional indices.
     * ConstExpr version for static rank tensors.
     * **/
    template <size_t N>
    [[nodiscard]] constexpr TensorIndexStatic<N> UnflattenIndex(const TensorStrideStatic<N> &strides, TensorIndex flatIndex) noexcept {
        TensorIndexStatic<N> indices{};

        if constexpr (N > 0) {
            for (size_t i = 0; i < N; ++i) {
                indices[i] = flatIndex / strides[i];
                flatIndex %= strides[i];
            }
        }
        return indices;
    }

    /** 
     * @brief Unflattens a single-dimensional index into multi-dimensional indices using the provided strides.
     * @param strides The strides of the tensor.
     * @param flatIndex The single-dimensional index to unflatten.
     * @return The unflattened multi-dimensional indices.
     * Runtime version for dynamic rank tensors.
     * **/
    [[nodiscard]] inline TensorIndexDynamic UnflattenIndex(const TensorStrideDynamic &strides, TensorIndex flatIndex) noexcept {
        TensorSize N = strides.size();
        TensorIndexDynamic indices(N);
        for (size_t i = 0; i < N; ++i) {
            indices[i] = flatIndex / strides[i];
            flatIndex %= strides[i];
        }
        return indices;
    }

    [[nodiscard]] inline void NextReverse(std::vector<size_t> &vec) noexcept {
        size_t left = 0;
        size_t right = vec.size() - 1;
        while (left < right) {
            std::swap(vec[left], vec[right]);
            ++left;
            --right;
        }
    }
}