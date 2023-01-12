#ifndef __POSTPROCESS_H
#define __POSTPROCESS_H

#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>

// struct AffineMatrix{
//     float value[6];
// };


  void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, int ckpt,float confidence_threshold, 
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_objects, cudaStream_t stream
    );
#endif  // __POSTPROCESS_H