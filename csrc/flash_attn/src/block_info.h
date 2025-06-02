/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>  // Varlen=!Is_even_MN
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
        , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
        // 矩阵KV实际的序列长度=(是否定义实际使用的KV长度 ? 际使用的KV长度-起始偏移 : KVCache长度+新产生的矩阵KV的序列长度)
        , actual_seqlen_k(params.seqused_k ? (params.seqused_k[bidb] - leftpad_k) : (seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)))
        {
        }

    // 当前线程块batch_idx下矩阵Q的起始偏移
    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        // 是否不使用累计序列长度(不是变长序列) ? batch_idx * batch_stride : 当前序列起始位置 * seqlen_stride
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    // 当前线程块batch_idx下矩阵KV的起始偏移
    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
    }

    // 当前batch_idx的矩阵Q累计的序列长度(起始偏移)
    const int sum_s_q;
    // 当前batch_idx的矩阵KV累计的序列长度
    const int sum_s_k;
    // 当前batch_idx的矩阵Q的实际序列长度
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    // 当前batch_idx的矩阵KV的起始偏移
    const int leftpad_k;
    // 当前batch_idx的KVCache的序列长度
    const int seqlen_k_cache;
    // 当前batch_idx的矩阵KV实际的序列长度(包括新产生的)
    const int actual_seqlen_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
