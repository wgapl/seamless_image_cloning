#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"
#define MAXTHREADS 1024


__global__ void computeMask(const uchar4* const d_sourceImg,
			    unsigned char* d_mask,
			    const size_t srcSize)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= srcSize) return;
  
  if (d_sourceImg[myId].x + d_sourceImg[myId].y + d_sourceImg[myId].z < 3 * 255)
    {
      d_mask[myId] = 1;
    }
  else
    {
      d_mask[myId] = 0;
    }

}

__global__ void oneStep(const unsigned char* const d_dstImg,
			const unsigned char* const d_strictInterior,
			const unsigned char* const d_border,
			const uint2* const d_interiorPixelList,
			const size_t numColsSource,
			const float* const f,
			const float* const g,
			float* const f_next,
			const size_t nInterior)
{

  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId >= nInterior) return;

  float blendedSum = 0.f;
  float borderSum = 0.f;

  uint2 coord = d_interiorPixelList[myId];
  unsigned int offset = coord.x * numColsSource + coord.y;

  // process all 4 neighbor pixels. If the pixel is interior then we
  // add the previousf, otherwise we add the value of the destination
  // image (our boundary conditions).
  
  // Pixel to the left of our interior point:
  if (d_strictInterior[offset -1]) {
    blendedSum += f[offset-1];
  }
  else {
    borderSum += d_dstImg[offset-1];
  }

  // Pixel to the right:
  if (d_strictInterior[offset+1]) {
    blendedSum += f[offset+1];
  }
  else {
    borderSum += d_dstImg[offset+1];
  }

  // Pixel above:
  if (d_strictInterior[offset-numColsSource]){
    blendedSum += f[offset - numColsSource];
  }
  else{
    borderSum += d_dstImg[offset - numColsSource];
  }

  // Pixel below:
  if (d_strictInterior[offset+numColsSource]){
    blendedSum += f[offset+numColsSource];
  }
  else{
    borderSum += d_dstImg[offset+numColsSource];
  }

  float f_next_val = (blendedSum + borderSum + g[offset])/ 4.f;

  //Clip to [0,255].
  f_next[offset] = fminf(255.f, fmaxf(0.f, f_next_val));

}
__global__ void computeBorderInterior(const unsigned char* const d_mask,
				      unsigned char* d_border,
				      unsigned char* d_strictInterior,
				      const size_t numRowsSource,
				      const size_t numColsSource,
				      const size_t srcSize)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId >= srcSize) return;

  // Find the row and column of this pixel.
  int r = myId / numColsSource;
  int c = myId % numColsSource;
  // Exclude any pixels that might be on the border of the source image
  if ((r == 0) || (c == 0) || (r == numRowsSource-1) || (c == numColsSource-1))
    {
      d_border[r * numColsSource + c] = 0;
      d_strictInterior[r * numColsSource + c] = 0;
      return;
    }

  if (d_mask[r * numColsSource + c])
    {
      if (d_mask[(r-1) * numColsSource + c] 
	  && d_mask[(r+1) * numColsSource + c] 
	  && d_mask[r * numColsSource + c - 1]
	  && d_mask[r * numColsSource + c + 1])
	{
	  // The pixel is in the strict interior
	  d_strictInterior[r * numColsSource + c] = 1;
	  d_border[r * numColsSource + c] = 0;
	}
      else
	{
	  // The pixel is on the border
	  d_strictInterior[r * numColsSource + c] = 0;
	  d_border[r * numColsSource + c] = 1;
	}
    }
  else
    {
      // The pixel is neither in the interior or on the border.
      d_strictInterior[r * numColsSource + c] = 0;
      d_border[r * numColsSource + c] = 0;
    }
}


__global__ void histogram_interior(const unsigned char* const d_strictInterior,
				   unsigned int* d_interiorCount,
				   const size_t srcSize)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  
  if (myId >= srcSize) return;

  if (d_strictInterior[myId])
    {
      atomicAdd(&(d_interiorCount[0]),1);
    }

}

__global__ void separateChannels(const uchar4* const d_sourceImg,
				 unsigned char* d_red,
				 unsigned char* d_blue,
				 unsigned char* d_green,
				 const size_t sizeSrc)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId >= sizeSrc) return;

  d_red[myId] = d_sourceImg[myId].x;
  d_blue[myId] = d_sourceImg[myId].y;
  d_green[myId] = d_sourceImg[myId].z;

}



__global__ void computeGTerm(const unsigned char* const d_channel,
			     float* d_g,
			     const uint2* const d_interiorPixelList,
			     const size_t numColsSource,
			     const size_t nInterior)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= nInterior) return;


  uint2 coord = d_interiorPixelList[myId];
  unsigned int offset = coord.x * numColsSource + coord.y;
  //cuPrintf("%d, %d\n", coord.x, coord.y);
  float sum = 4.f * d_channel[offset];

  sum -= (float)d_channel[offset-1] + (float)d_channel[offset+1];
  sum -= (float)d_channel[offset + numColsSource] + (float)d_channel[offset - numColsSource];
  
  d_g[offset] = sum;

}


__global__ void gpuCopyBuffers(float* d_dst_1,
			       float* d_dst_2,
			       const unsigned char* const d_src,
			       const size_t srcSize)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= srcSize) return;

  d_dst_1[myId] = d_src[myId];
  d_dst_2[myId] = d_src[myId];
}
  
__global__ void gpuSwapper(float* d_f,
			   float* d_g,
			   const size_t sizeSrc)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= sizeSrc) return;

  float f_val = d_f[myId];
  float g_val = d_g[myId];
  __syncthreads();

  d_f[myId] = g_val;
  d_g[myId] = f_val;
}
			     
__global__ void copyInterior(uchar4* d_blendedImg,
			     const uint2* const d_interiorPixelList,
			     const float* const d_blendedValsRed_2,
			     const float* const d_blendedValsBlue_2,
			     const float* const d_blendedValsGreen_2,
			     const size_t numColsSource,
			     const size_t nInterior)
{
  int myId = threadIdx.x + blockDim.x * blockIdx.x;

  if (myId >= nInterior) return;

  uint2 coord = d_interiorPixelList[myId];

  unsigned int offset = coord.x * numColsSource + coord.y;

  d_blendedImg[offset].x = d_blendedValsRed_2[offset];
  d_blendedImg[offset].y = d_blendedValsBlue_2[offset];
  d_blendedImg[offset].z = d_blendedValsGreen_2[offset];

}


void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  // Copy h_sourceImg over to the GPU.
  size_t srcSize = numRowsSource * numColsSource;
  uchar4* d_sourceImg;

  checkCudaErrors(cudaMalloc((void**) &d_sourceImg,
			     srcSize*sizeof(uchar4)));
			     
  checkCudaErrors(cudaMemcpy(d_sourceImg,
			     h_sourceImg,
			     srcSize*sizeof(uchar4),
			     cudaMemcpyHostToDevice));

  // Copy h_destImg over to GPU.
  uchar4* d_destImg = new uchar4[srcSize];
  
  checkCudaErrors(cudaMalloc((void**) &d_destImg,
			     srcSize*sizeof(uchar4)));

  checkCudaErrors(cudaMemcpy(d_destImg,
			     h_destImg,
			     srcSize*sizeof(uchar4),
			     cudaMemcpyHostToDevice));

  // Create a mask from the source image. If the pixel is pure white,
  // then d_mask[myId] = 0, otherwise d_mask[myId] = 1.
  unsigned char* d_mask;
  checkCudaErrors(cudaMalloc((void**) &d_mask,
			     srcSize*sizeof(unsigned char)));

  int nBlocks = srcSize / MAXTHREADS + 1;
  std::cout << nBlocks << std::endl;
  
  computeMask <<< nBlocks, MAXTHREADS >>> (d_sourceImg,
					   d_mask,
					   srcSize);

  // Next we need to find the border and strict interior of the mask.
  unsigned char* d_border;
  unsigned char* d_strictInterior;

  checkCudaErrors(cudaMalloc((void**) &d_border,
			     srcSize*sizeof(unsigned char)));

  checkCudaErrors(cudaMalloc((void**) &d_strictInterior,
			     srcSize*sizeof(unsigned char)));

  computeBorderInterior <<< nBlocks, MAXTHREADS >>> (d_mask,
						     d_border,
						     d_strictInterior,
						     numRowsSource,
						     numColsSource,
						     srcSize);

  // We do a parallel histogram to count of the number of elements of
  // d_strictInterior.
  unsigned int* d_interiorCount;
  checkCudaErrors(cudaMalloc((void**) &d_interiorCount,
			     sizeof(unsigned int)));
  checkCudaErrors(cudaMemset(d_interiorCount,
			     0,
			     sizeof(unsigned int)));

  histogram_interior <<< nBlocks, MAXTHREADS >>> (d_strictInterior,
						  d_interiorCount,
						  srcSize);

  // We copy the output back over to the host for use on the CPU.
  unsigned int h_interiorCount[1];
  checkCudaErrors(cudaMemcpy(h_interiorCount,
			     d_interiorCount,
			     sizeof(unsigned int),
			     cudaMemcpyDeviceToHost));

  unsigned int nInterior = h_interiorCount[0];
  std::cout << nInterior << std::endl;

  // We create a uint2 array to keep track of the indices of each
  // pixel in the strict interior.
  uint2 h_interiorList[nInterior];

  unsigned char h_border[srcSize];
  unsigned char h_mask[srcSize];

  // We copy d_strictInterior over to the host to create
  // h_interiorList and define it on the CPU.
  unsigned char h_strictInterior[srcSize];

  checkCudaErrors(cudaMemcpy(h_mask,
			     d_mask,
			     srcSize*sizeof(unsigned char),
			     cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_border,
                             d_border,
                             srcSize*sizeof(unsigned char),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(h_strictInterior,
                             d_strictInterior,
                             srcSize*sizeof(unsigned char),
                             cudaMemcpyDeviceToHost));

  // We have to go through every pixel in h_strictInterior
  int j = 0;
  for (int i = 0; i < srcSize; i++)
    {
      // Keep track of the row and column of each pixel.
      int r = i / numColsSource;
      int c = i % numColsSource;
      // If the pixel is in the strict interior...
      if (h_strictInterior[i])
	{
	  // Put a uint2 into the interior list of the pixel's
	  // location.
	  uint2 vals = make_uint2(r,c);
	  //std::cout << vals.x << "\t" << vals.y << std::endl;
	  h_interiorList[j] = vals;
	  j++;
	}
    }

  // Now copy h_interiorList over to d_interiorPixelList

  uint2* d_interiorPixelList;

  checkCudaErrors(cudaMalloc((void**) &d_interiorPixelList,
			     nInterior*sizeof(uint2)));

  checkCudaErrors(cudaMemcpy(d_interiorPixelList,
			     h_interiorList,
			     nInterior*sizeof(uint2),
			     cudaMemcpyHostToDevice));


  // Now separate out the source image into three channels.
  unsigned char* d_red_src = new unsigned char[srcSize];
  unsigned char* d_blue_src = new unsigned char[srcSize];
  unsigned char* d_green_src = new unsigned char[srcSize];

  checkCudaErrors(cudaMalloc((void**) &d_red_src,
			     srcSize*sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc((void**) &d_blue_src,
			     srcSize*sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc((void**) &d_green_src,
			     srcSize*sizeof(unsigned char)));

  separateChannels <<< nBlocks, MAXTHREADS >>> (d_sourceImg,
						d_red_src,
						d_blue_src,
						d_green_src,
						srcSize);

  // Now separate the channels of the destination image.
  unsigned char* d_red_dst = new unsigned char[srcSize];
  unsigned char* d_blue_dst = new unsigned char[srcSize];
  unsigned char* d_green_dst = new unsigned char[srcSize];

  checkCudaErrors(cudaMalloc((void**) &d_red_dst,
                             srcSize*sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc((void**) &d_blue_dst,
                             srcSize*sizeof(unsigned char)));
  checkCudaErrors(cudaMalloc((void**) &d_green_dst,
                             srcSize*sizeof(unsigned char)));

  
  separateChannels <<< nBlocks, MAXTHREADS >>> (d_destImg,
                                                d_red_dst,
                                                d_blue_dst,
                                                d_green_dst,
                                                srcSize);

  // Now we compute the g term since we only need to do so once.
  
  float* d_g_red = new float[srcSize];
  float* d_g_blue = new float[srcSize];
  float* d_g_green = new float[srcSize];

  checkCudaErrors(cudaMalloc((void**) &d_g_red,
			     srcSize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_g_blue,
			     srcSize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_g_green,
			     srcSize*sizeof(float)));

  checkCudaErrors(cudaMemset(d_g_red,
			     0,
			     srcSize*sizeof(float)));
  checkCudaErrors(cudaMemset(d_g_blue,
			     0,
			     srcSize*sizeof(float)));
  checkCudaErrors(cudaMemset(d_g_green,
			     0,
			     srcSize*sizeof(float)));

  int nBlocksInterior = nInterior / MAXTHREADS + 1;

  computeGTerm <<< nBlocksInterior, MAXTHREADS >>> (d_red_src,
						    d_g_red,
						    d_interiorPixelList,
						    numColsSource,
						    nInterior);

  
  computeGTerm <<< nBlocksInterior, MAXTHREADS >>> (d_blue_src,
                                                    d_g_blue,
                                                    d_interiorPixelList,
                                                    numColsSource,
                                                    nInterior);


  computeGTerm <<< nBlocksInterior, MAXTHREADS >>> (d_green_src,
                                                    d_g_green,
                                                    d_interiorPixelList,
                                                    numColsSource,
                                                    nInterior);


  // Declare double buffers for all three channels.
  float* d_blendedValsRed_1 = new float[srcSize];
  float* d_blendedValsRed_2 = new float[srcSize];

  float* d_blendedValsBlue_1 = new float[srcSize];
  float* d_blendedValsBlue_2 = new float[srcSize];

  float* d_blendedValsGreen_1 = new float[srcSize];
  float* d_blendedValsGreen_2 = new float[srcSize];

  // Allocate memory for buffers on the GPU.
  checkCudaErrors(cudaMalloc((void**) &d_blendedValsRed_1,
			     srcSize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_blendedValsRed_2,
                             srcSize*sizeof(float)));

  checkCudaErrors(cudaMalloc((void**) &d_blendedValsBlue_1,
                             srcSize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_blendedValsBlue_2,
                             srcSize*sizeof(float)));

  checkCudaErrors(cudaMalloc((void**) &d_blendedValsGreen_1,
                             srcSize*sizeof(float)));
  checkCudaErrors(cudaMalloc((void**) &d_blendedValsGreen_2,
                             srcSize*sizeof(float)));


  // Copy over from d_*_src into the buffers. Launching 3 kernels

  gpuCopyBuffers <<< nBlocks, MAXTHREADS >>> (d_blendedValsRed_1,
					      d_blendedValsRed_2,
					      d_red_src,
					      srcSize);

  gpuCopyBuffers <<< nBlocks, MAXTHREADS >>> (d_blendedValsBlue_1,
                                              d_blendedValsBlue_2,
                                              d_blue_src,
                                              srcSize);

  gpuCopyBuffers <<< nBlocks, MAXTHREADS >>> (d_blendedValsGreen_1,
                                              d_blendedValsGreen_2,
                                              d_green_src,
                                              srcSize);

  // Now do 800 jacobi iterations on the red channel.
  const size_t numIterations = 800;
  for (size_t i = 0; i < numIterations; ++i) {

    oneStep <<< nBlocksInterior, MAXTHREADS >>> (d_red_dst, 
						 d_strictInterior,
						 d_border,
						 d_interiorPixelList,
						 numColsSource,
						 d_blendedValsRed_1,
						 d_g_red,
						 d_blendedValsRed_2,
						 nInterior);
   
    
    gpuSwapper <<< nBlocks, MAXTHREADS >>> (d_blendedValsRed_1, 
					    d_blendedValsRed_2,
					    srcSize);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    
  }

  // 800 more steps on the blue channel.
  for (size_t i = 0; i < numIterations; ++i) {

    oneStep <<< nBlocksInterior, MAXTHREADS >>> (d_blue_dst,
                                                 d_strictInterior,
                                                 d_border,
                                                 d_interiorPixelList,
                                                 numColsSource,
                                                 d_blendedValsBlue_1,
                                                 d_g_blue,
                                                 d_blendedValsBlue_2,
                                                 nInterior);

    gpuSwapper <<< nBlocks, MAXTHREADS >>> (d_blendedValsBlue_1,
                                            d_blendedValsBlue_2,
                                            srcSize);
  }

  for (size_t i = 0; i < numIterations; ++i) {
    oneStep <<< nBlocksInterior, MAXTHREADS >>> (d_green_dst,
                                                 d_strictInterior,
                                                 d_border,
                                                 d_interiorPixelList,
                                                 numColsSource,
                                                 d_blendedValsGreen_1,
                                                 d_g_green,
                                                 d_blendedValsGreen_2,
                                                 nInterior);

    gpuSwapper <<< nBlocks, MAXTHREADS >>> (d_blendedValsGreen_1,
                                            d_blendedValsGreen_2,
                                            srcSize);

  }


  // Copy Destination image onto output.
  memcpy(h_blendedImg, h_destImg, srcSize*sizeof(uchar4));

  // create array for output on the GPU.
  uchar4* d_blendedImg = new uchar4[srcSize];

  checkCudaErrors(cudaMalloc((void**) &d_blendedImg,
			     srcSize*sizeof(uchar4)));

  // Copy h_blendedImg over to GPU.
  checkCudaErrors(cudaMemcpy(d_blendedImg,
			     h_blendedImg,
			     srcSize*sizeof(uchar4),
			     cudaMemcpyHostToDevice));

  
  // Launch copyInterior kernel.
  // Copy results of 800 jacobi iterations into the output image
  copyInterior <<< nBlocksInterior, MAXTHREADS >>> (d_blendedImg,
						    d_interiorPixelList,
						    d_blendedValsRed_2,
						    d_blendedValsBlue_2,
						    d_blendedValsGreen_2,
						    numColsSource,
						    nInterior);
  
  
  checkCudaErrors(cudaMemcpy(h_blendedImg,
			     d_blendedImg,
			     srcSize*sizeof(uchar4),
			     cudaMemcpyDeviceToHost));


}
