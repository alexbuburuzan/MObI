#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

/* Force the pre-processor to ignore the 128-bit definitions */
#undef __SIZEOF_INT128__
#define __signed__ signed

#endif