#pragma once

#include <sys/types.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct bitcompContext;
  typedef struct bitcompContext *bitcompHandle_t;

  typedef enum bitcompResult_t
  {
    BITCOMP_SUCCESS = 0,
    BITCOMP_INVALID_PARAMETER = -1,
    BITCOMP_INVALID_COMPRESSED_DATA = -2,
    BITCOMP_INVALID_ALIGNMENT = -3,
    BITCOMP_UNKNOWN_ERROR = -4
  } bitcompResult_t;

  typedef enum bitcompDataType_t
  {
    // Integral types for lossless compression
    BITCOMP_UNSIGNED_8BIT = 0,
    BITCOMP_SIGNED_8BIT,
    BITCOMP_UNSIGNED_16BIT,
    BITCOMP_SIGNED_16BIT,
    BITCOMP_UNSIGNED_32BIT,
    BITCOMP_SIGNED_32BIT,
    BITCOMP_UNSIGNED_64BIT,
    BITCOMP_SIGNED_64BIT,
    // Floating point types used for lossy compression
    BITCOMP_FP16_DATA,
    BITCOMP_FP32_DATA,
    BITCOMP_FP64_DATA
  } bitcompDataType_t;

  typedef enum bitcompMode_t
  {
    // Compression mode, lossless or lossy
    BITCOMP_LOSSLESS = 0,
    BITCOMP_LOSSY_FP_TO_SIGNED,
    BITCOMP_LOSSY_FP_TO_UNSIGNED
  } bitcompMode_t;

  typedef enum bitcompAlgorithm_t
  {
    BITCOMP_DEFAULT_ALGO = 0, // Default algorithm
    BITCOMP_SPARSE_ALGO = 1   // Recommended for very sparse data (lots of zeros)
  } bitcompAlgorithm_t;

  //***********************************************************************************************
  // Plan creation and destruction

  /**
   * @brief Create a bitcomp plan for compression and decompression, lossy or lossless.
   *
   * The lossless compression can be used on any data type, viewed as integral type.
   * Choosing the right integral type will have an effect on the compression ratio.
   *
   * Lossy compression:
   * The lossy compression is only available for floating point data types, and is based
   * on a quantization of the floating point values to integers.
   * The floating point values are divided by the delta provided during the compression, and converted
   * to integers. These integers are then compressed with a lossless encoder.
   * Values that would overflow during quantization (e.g. large input values and a very small delta),
   * as well as NaN, +Inf, -Inf will be handled correctly by the compression.
   * The integers can be either signed or unsigned.
   *
   * The same plan can be used on several devices or on the host, but associating the plan
   * with a stream, or turning on remote compression acceleration will make a plan device-specific.
   * Using a plan concurrently on more than one device is not supported.
   * 
   * @param handle (output) Handle created
   * @param n (input) size of the uncompressed data in bytes
   * @param dataType (input) Datatype of the uncompressed data.
   * @param mode (input) Compression mode, lossless or lossy to signed / lossy to unsigned
   * @param algo (input) Which compression algorithm to use
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompCreatePlan(bitcompHandle_t *handle,    
                                    size_t n,                   
                                    bitcompDataType_t dataType, 
                                    bitcompMode_t mode,         
                                    bitcompAlgorithm_t algo);   

  /**
   * @brief Create a handle from existing compressed data.
   * 
   * @param handle (output) Handle
   * @param data  (input) Pointer to the compressed data, from which all the handle parameters will be extracted.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompCreatePlanFromCompressedData(bitcompHandle_t *handle, 
                                                      const char *data);       

  /**
   * @brief Destroy an existing bitcomp handle
   * 
   * @param handle (input/output) Handle to destroy
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompDestroyPlan(bitcompHandle_t handle);

  //***********************************************************************************************
  // Modification of plan attributes

  /**
   * @brief Associate a bitcomp handle to a stream. All the subsequent operations will be done in the stream.
   * 
   * @param handle (input) Bitcomp handle
   * @param stream (input) Stream to use.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompSetStream(bitcompHandle_t handle, cudaStream_t stream);

  /**
   * @brief Turn on compression acceleration when the compressed output is not in the global memory
   * of the device running the compression (e.g. host pinned memory, or another device's memory)
   * This is optional and only affects the performance.
   * NOTE: This makes the handle become device-specific. A plan that has this acceleration turned on
   * should always be used on the same device.
   * 
   * @param handle (input) Bitcomp handle
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompAccelerateRemoteCompression(bitcompHandle_t handle);

  //***********************************************************************************************
  // Compression and decompression on the device

  /**
   * @brief Compression for FP16 (half) data, running asynchronously on the device.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the uncompressed data. Must be accessible from the device.
   * @param output (output) Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
   * @param delta (input) Delta used for the integer quantization of the data.
   * The maximum error between the uncompressed data and the original data should be <= delta.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompCompressLossy_fp16(const bitcompHandle_t handle,
                                            const half *input,           
                                            char *output,                
                                            half delta);                 

  /**
   * @brief Compression for 32-bit floating point data, running asynchronously on the device.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the uncompressed data. Must be accessible from the device.
   * @param output (output) Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
   * @param delta (input) Delta used for the integer quantization of the data.
   * The maximum error between the uncompressed data and the original data should be <= delta.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompCompressLossy_fp32(const bitcompHandle_t handle,
                                            const float *input,          
                                            char *output,                
                                            float delta);                

  /**
   * @brief Compression for 64-bit floating point data, running asynchronously on the device.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the uncompressed data. Must be accessible from the device.
   * @param output (output) Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
   * @param delta (input) Delta used for the integer quantization of the data.
   * The maximum error between the uncompressed data and the original data should be <= delta.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompCompressLossy_fp64(const bitcompHandle_t handle,
                                            const double *input,         
                                            char *output,                
                                            double delta);               

  // bitcompCompressLossless: Lossless compression of integral data types
  bitcompResult_t bitcompCompressLossless(const bitcompHandle_t handle,
                                          const void *input,           
                                          char *output);               

  /**
   * @brief Decompression, running asynchronously on the device.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
   * @param output (output) Pointer to where the uncompressed data will be written.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompUncompress(const bitcompHandle_t handle,
                                    const char *input,           
                                    void *output);               

  /**
   * @brief Partial decompression, running asynchronously on the device.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the compressed data. Must be accessible from the device and 64-bit aligned.
   * @param output (output) Pointer to where the partial uncompressed data will be written.
   * @param start (input) Offset in bytes relative to the original uncompressed size where to start decompressing.
   * @param length (input) Length in bytes of the partial decompression.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompPartialUncompress(const bitcompHandle_t handle,
                                           const char *input,           
                                           void *output,                
                                           off_t start,                 
                                           size_t length);              

  //***********************************************************************************************
  // Compression and decompression on the host

  /**
   * @brief Lossy compression for FP16 (half) data, running on the host processor. This call is blocking.
   * If a non-NULL stream was set in the handle, this call will synchronize the stream
   * before compressing the data.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the uncompressed data. Must be accessible from the host.
   * @param output (output) Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
   * @param delta (input) Delta used for the integer quantization of the data.
   * The maximum error between the uncompressed data and the original data should be <= delta.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompHostCompressLossy_fp16(const bitcompHandle_t handle,
                                                const half *input,
                                                char *output,
                                                half delta);

  /**
   * @brief Lossy compression for 32-bit floats, running on the host processor. This call is blocking.
   * If a non-NULL stream was set in the handle, this call will synchronize the stream
   * before compressing the data.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the uncompressed data. Must be accessible from the host.
   * @param output (output) Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
   * @param delta (input) Delta used for the integer quantization of the data.
   * The maximum error between the uncompressed data and the original data should be <= delta.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompHostCompressLossy_fp32(const bitcompHandle_t handle,
                                                const float *input,          
                                                char *output,                
                                                float delta);                

  /**
   * @brief Lossy compression for 64-bit floats, running on the host processor. This call is blocking.
   * If a non-NULL stream was set in the handle, this call will synchronize the stream
   * before compressing the data.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the uncompressed data. Must be accessible from the host.
   * @param output (output) Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
   * @param delta (input) Delta used for the integer quantization of the data.
   * The maximum error between the uncompressed data and the original data should be <= delta.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompHostCompressLossy_fp64(const bitcompHandle_t handle,
                                                const double *input,         
                                                char *output,                
                                                double delta);               

  /**
   * @brief Lossless compression (integral datatypes), running on the host processor. This call is blocking.
   * If a non-NULL stream was set in the handle, this call will synchronize the stream
   * before compressing the data.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the uncompressed data. Must be accessible from the host.
   * @param output (output) Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompHostCompressLossless(const bitcompHandle_t handle,
                                              const void *input,           
                                              char *output);               

  //bitcompHostUncompress: Decompression of compressed data
  /**
   * @brief Decompression, running on the host processor. This call is blocking.
   * If a non-NULL stream was set in the handle, this call will synchronize the stream
   * before decompressing the data.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
   * @param output (output) Pointer to the uncompressed data. Must be accessible from the host.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompHostUncompress(const bitcompHandle_t handle,
                                        const char *input,           
                                        void *output);               

  /**
   * @brief Partial decompression, running on the host processor. This call is blocking.
   * If a non-NULL stream was set in the handle, this call will synchronize the stream
   * before decompressing the data.
   * 
   * @param handle (input) Bitcomp handle
   * @param input (input) Pointer to the compressed data. Must be accessible from the host and 64-bit aligned.
   * @param output (output) Pointer to where the partial uncompressed data will be written.
   * @param start (input) Offset in bytes relative to the original uncompressed size where to start decompressing.
   * @param length (input) Length in bytes of the partial decompression.
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompHostPartialUncompress(const bitcompHandle_t handle,
                                               const char *input,           
                                               void *output,                
                                               off_t start,                 
                                               size_t length);              

  // *******************************************************************************************************************
  // Utilities

  /**
   * @brief Query the maximum size (worst case scenario) that the compression could
   * generate given an input size.
   * 
   * @param nbytes (input) Size of the uncompressed data, in bytes
   * @return Returns the maximum size of the compressed data, in bytes.
   */
  size_t bitcompMaxBuflen(size_t nbytes);

  /**
   * @brief Query the compressed size from a compressed buffer.
   * The pointers don't have to be device-accessible. This is a blocking call.
   * The compression must have completed before calling this function.
   * 
   * @param compressedData (input) Pointer to the compressed data
   * @param size (output) Size of the compressed data, in bytes
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompGetCompressedSize(const char *compressedData, size_t *size);

  /**
   * @brief Query the compressed size from a compressed buffer, asynchronously.
   * Both pointers must be device-accessible.
   * 
   * @param compressedData (input) Pointer to the compressed data
   * @param size (output) Size of the compressed data, in bytes
   * @param stream (input) Stream for asynchronous operation
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompGetCompressedSizeAsync(const char *compressedData, size_t *size, cudaStream_t stream);

  /**
   * @brief Query the uncompressed size from a compressed buffer
   * 
   * @param compressedData (input) Pointer to the compressed data buffer,
   * The pointer doesn't have to be device-accessible.
   * @param size (output) Size of the uncompressed data, in bytes
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompGetUncompressedSize(const char *compressedData, size_t *size);

  /**
   * @brief Query the uncompressed size from a handle
   * 
   * @param handle (input) handle
   * @param bytes (output) Size in bytes of the uncompressed data
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompGetUncompressedSizeFromHandle(const bitcompHandle_t handle, size_t *bytes);

  /**
   * @brief Query the uncompressed datatype from a handle
   * 
   * @param handle (input) handle
   * @param dataType (output) Data type of the uncompressed data
   * @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompGetDataTypeFromHandle(const bitcompHandle_t handle, bitcompDataType_t *dataType);

  /** @brief: Query compressed data information
   *  @param compressedData (input) : Compressed data pointer. Doesn't have to be device-accessible
   *  @param compressedDataSize (input): Size of the compressed buffer, (output): actual size of the compressed data
   *  If the size of the compressed buffer is smaller than the actual size of the compressed data,
   *  BITCOMP_INVALID_PARAMETER will be returned.
   *  @param uncompressedSize (output): The size of the uncompressed data in bytes
   *  @param dataType (output): The type of the compressed data.
   *  @param mode (output): Compression mode (lossy or lossless)
   *  @param algo (output): Bitcomp alogorithm used (default, or sparse)
   *  @return Returns BITCOMP_SUCCESS if successful, or an error
   */
  bitcompResult_t bitcompGetCompressedInfo(const char *compressedData,
                                           size_t *compressedDataSize,
                                           size_t *uncompressedSize,
                                           bitcompDataType_t *dataType,
                                           bitcompMode_t *mode,
                                           bitcompAlgorithm_t *algo);

#ifdef __cplusplus
}
#endif
