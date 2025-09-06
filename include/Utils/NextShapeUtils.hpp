#pragma once

#include "../Core/TensorMetadata.hpp"

using TensorMetadata = NextMetadata::TensorMetadata;

namespace NextShapeUtils
{
    /** 
     * @brief Computes Slice of a tensor given its metadata and start/end indices.
     * @param Metadata The metadata of the tensor.
     * @param startIndices The starting indices for the slice.
     * @param endIndices The ending indices for the slice.
     * @return A new TensorMetadata object representing the sliced tensor.
     * @throws std::invalid_argument if start and end indices do not match the tensor's rank.
     * @throws std::out_of_range if start or end indices are out of bounds
     * **/
    inline TensorMetadata NextSlice(const TensorMetadata &Metadata, const TensorIndexDynamic &startIndices, const TensorIndexDynamic &endIndices) {
        // Validate start and end indices
        if (startIndices.size() != Metadata.GetShape().size() || endIndices.size() != Metadata.GetShape().size()) {
            throw std::invalid_argument("Start and end indices must match the tensor's rank.");
        }

        // Compute new shape and strides
        TensorShapeDynamic newShape;
        TensorStrideDynamic newStrides;
        for (size_t i = 0; i < Metadata.GetShape().size(); ++i) {
            if (startIndices[i] >= endIndices[i] || startIndices[i] >= Metadata.GetShape()[i] || endIndices[i] > Metadata.GetShape()[i]) {
                throw std::out_of_range("Start or end indices are out of bounds.");
            }
            newShape.push_back(endIndices[i] - startIndices[i]);
            newStrides.push_back(Metadata.GetStrides()[i]);
        }

        // Create new metadata with the sliced shape and strides
        return TensorMetadata(newShape, Metadata.GetOffset());

    }

    /** 
     * @brief Reshapes a tensor given its metadata and a new shape.
     * @param Metadata The metadata of the tensor.
     * @param newShape The new shape for the tensor.
     * @return A new TensorMetadata object representing the reshaped tensor.
     * @throws std::invalid_argument if the total size does not match.
     * **/
    inline TensorMetadata NextReshape(const TensorMetadata &Metadata, const TensorShapeDynamic &newShape) {
        // Validate that the total size remains the same
        size_t oldSize = Metadata.GetTotalSize();
        size_t newSize = 1;
        for (const auto& dim : newShape) {
            newSize *= dim;
        }
        if (oldSize != newSize) {
            throw std::invalid_argument("New shape must have the same total size as the original tensor.");
        }

        // Compute new strides
        TensorStrideDynamic newStrides = NextUtils::ComputeStrides(newShape);

        // Create new metadata with the reshaped shape and computed strides
        return TensorMetadata(newShape, Metadata.GetOffset());
    }

    /**
     * @brief Permutes the dimensions of a tensor.
     * @param metadata The metadata of the tensor.
     * @param permutation The new order of dimensions.
     * @return A new TensorMetadata object representing the permuted tensor.
     * @throws std::invalid_argument if the permutation is invalid.
     * **/
    inline TensorMetadata NextPermute(const TensorMetadata &Metadata, TensorIndexDynamic permutation = {}) {
        TensorShapeDynamic originalShape = Metadata.GetShape();
        TensorStrideDynamic originalStrides = Metadata.GetStrides();

        if (permutation.empty()) {
            // Case 1: Standard transpose (reverses the dimensions)
            TensorShapeDynamic newShape = originalShape;
            TensorStrideDynamic newStrides = originalStrides;
            NextUtils::NextReverse(newShape);
            NextUtils::NextReverse(newStrides);
            return TensorMetadata(newShape, newStrides, Metadata.GetOffset());
        }

        // Case 2: Custom permutation
        if (permutation.size() != originalShape.size()) {
            throw std::invalid_argument("Permutation vector must have the same number of elements as the tensor's rank.");
        }

        TensorShapeDynamic newShape(originalShape.size());
        TensorStrideDynamic newStrides(originalStrides.size());

        for (size_t i = 0; i < permutation.size(); ++i) {
            size_t newIndex = permutation[i];
            // You can add more checks here to ensure newIndex is valid and not a duplicate
            newShape[i] = originalShape[newIndex];
            newStrides[i] = originalStrides[newIndex];
        }

        return TensorMetadata(newShape, newStrides, Metadata.GetOffset());
    }

    /**
     * @brief Transposes the dimensions of a tensor (alias for NextPermute with default behavior).
     * @param metadata The metadata of the tensor.
     * @return A new TensorMetadata object representing the transposed tensor.
     * **/
    inline TensorMetadata NextTranspose(const TensorMetadata &Metadata) {
        return NextPermute(Metadata);
    }

    //TODO: Review and Optimize Squeeze and UnSqueeze Functions
    /**
     * @brief Squeezes the dimensions of a tensor by removing single-dimensional entries.
     * @param metadata The metadata of the tensor.
     * @param axes The specific axes to squeeze. If empty, all single-dimensional axes are removed.
     * @return A new TensorMetadata object representing the squeezed tensor.
     * @throws std::invalid_argument if any specified axis is out of bounds or not of size 1.
     * **/
    inline TensorMetadata NextSqueeze(const TensorMetadata &Metadata, TensorIndexDynamic axes = {}) {
        // Check if valid axes are provided
        if (!axes.empty()) {
            TensorShapeDynamic newShape;
            TensorStrideDynamic newStrides;
            for (const auto &axis : axes) {
                if (axis >= Metadata.GetRank()) {
                    throw std::invalid_argument("Axis out of bounds.");
                }
                if (Metadata.GetShape()[axis] != 1) {
                    throw std::invalid_argument("Cannot squeeze axis that is not of size 1.");
                }
                // Proceed to create new shape and strides
                for (size_t i = 0; i < Metadata.GetRank(); i++) {
                    if (i == axis) continue; // Skip the specified axis
                    newShape.push_back(Metadata.GetShape()[i]);
                }
            }
            return TensorMetadata(newShape, Metadata.GetOffset());
        }
        else {
            TensorShapeDynamic newShape;
            TensorStrideDynamic newStrides;
            for (size_t i = 0; i < Metadata.GetRank(); i++) {
                if (Metadata.GetShape()[i] == 1) continue; // Skip single-dimensional axes
                newShape.push_back(Metadata.GetShape()[i]);
            }
            return TensorMetadata(newShape, Metadata.GetOffset());
        }
    }

    /**
     * @brief Unsqueezes the dimensions of a tensor by adding single-dimensional entries at specified axes.
     * @param metadata The metadata of the tensor.
     * @param axes The specific axes to unsqueeze.
     * @return A new TensorMetadata object representing the unsqueezed tensor.
     * @throws std::invalid_argument if any specified axis is out of bounds.
     * **/
    inline TensorMetadata NextUnsqueeze(const TensorMetadata &Metadata, const TensorIndexDynamic &axes) {
        TensorShapeDynamic newShape = Metadata.GetShape();
        TensorStrideDynamic newStrides = Metadata.GetStrides();

        for (const auto &axis : axes) {
            if (axis > newShape.size()) {
                throw std::invalid_argument("Axis out of bounds.");
            }
            newShape.insert(newShape.begin() + axis, 1);
        }
        return TensorMetadata(newShape, Metadata.GetOffset());
    }
}