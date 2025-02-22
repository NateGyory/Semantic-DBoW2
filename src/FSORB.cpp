/**
 * File: FSORB.cpp
 * Date: September 2022
 * Author: Nathaniel Gyory
 * Description: functions for semantic ORB descriptors
 * License: see the LICENSE.txt file
 *
 */

#include <vector>
#include <string>
#include <sstream>
#include <tuple>
#include <stdint.h>
#include <limits.h>

#include "FSORB.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FSORB::meanValue(const std::vector<FSORB::pDescriptor> &descriptors,
  FSORB::TDescriptor &mean)
{
  cv::Mat &mean_ref = mean.first;
  if(descriptors.empty())
  {
    mean_ref.release();
    return;
  }
  else if(descriptors.size() == 1)
  {
    mean_ref = (*descriptors[0]).first.clone();
  }
  else
  {
    vector<int> sum(FSORB::L * 8, 0);

    for(size_t i = 0; i < descriptors.size(); ++i)
    {
      const cv::Mat &d = (*descriptors[i]).first;
      const unsigned char *p = d.ptr<unsigned char>();

      for(int j = 0; j < d.cols; ++j, ++p)
      {
        if(*p & (1 << 7)) ++sum[ j*8     ];
        if(*p & (1 << 6)) ++sum[ j*8 + 1 ];
        if(*p & (1 << 5)) ++sum[ j*8 + 2 ];
        if(*p & (1 << 4)) ++sum[ j*8 + 3 ];
        if(*p & (1 << 3)) ++sum[ j*8 + 4 ];
        if(*p & (1 << 2)) ++sum[ j*8 + 5 ];
        if(*p & (1 << 1)) ++sum[ j*8 + 6 ];
        if(*p & (1))      ++sum[ j*8 + 7 ];
      }
    }

    mean_ref = cv::Mat::zeros(1, FSORB::L, CV_8U);
    unsigned char *p = mean_ref.ptr<unsigned char>();

    const int N2 = (int)descriptors.size() / 2 + descriptors.size() % 2;
    for(size_t i = 0; i < sum.size(); ++i)
    {
      if(sum[i] >= N2)
      {
        // set bit
        *p |= 1 << (7 - (i % 8));
      }

      if(i % 8 == 7) ++p;
    }
  }
}

// --------------------------------------------------------------------------

double FSORB::distance(const FSORB::TDescriptor &a,
  const FSORB::TDescriptor &b)
{
  // Bit count function got from:
  // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
  // This implementation assumes that a.cols (CV_8U) % sizeof(uint64_t) == 0

  const uint64_t *pa, *pb;
  pa = (a.first).ptr<uint64_t>(); // a & b are actually CV_8U
  pb = (b.first).ptr<uint64_t>();

  uint64_t v, ret = 0;
  for(size_t i = 0; i < (a.first).cols / sizeof(uint64_t); ++i, ++pa, ++pb)
  {
    v = *pa ^ *pb;
    v = v - ((v >> 1) & (uint64_t)~(uint64_t)0/3);
    v = (v & (uint64_t)~(uint64_t)0/15*3) + ((v >> 2) &
      (uint64_t)~(uint64_t)0/15*3);
    v = (v + (v >> 4)) & (uint64_t)~(uint64_t)0/255*15;
    ret += (uint64_t)(v * ((uint64_t)~(uint64_t)0/255)) >>
      (sizeof(uint64_t) - 1) * CHAR_BIT;
  }

  return static_cast<double>(ret);

  // // If uint64_t is not defined in your system, you can try this
  // // portable approach (requires DUtils from DLib)
  // const unsigned char *pa, *pb;
  // pa = a.ptr<unsigned char>();
  // pb = b.ptr<unsigned char>();
  //
  // int ret = 0;
  // for(int i = 0; i < a.cols; ++i, ++pa, ++pb)
  // {
  //   ret += DUtils::LUT::ones8bits[ *pa ^ *pb ];
  // }
  //
  // return ret;
}

// --------------------------------------------------------------------------

std::string FSORB::toString(const FSORB::TDescriptor &a)
{
  stringstream ss;
  const unsigned char *p = (a.first).ptr<unsigned char>();

  for(int i = 0; i < (a.first).cols; ++i, ++p)
  {
    ss << (int)*p << " ";
  }

  return ss.str();
}

// --------------------------------------------------------------------------

void FSORB::fromString(FSORB::TDescriptor &a, const std::string &s)
{
  (a.first).create(1, FSORB::L, CV_8U);
  unsigned char *p = (a.first).ptr<unsigned char>();

  stringstream ss(s);
  for(int i = 0; i < FSORB::L; ++i, ++p)
  {
    int n;
    ss >> n;

    if(!ss.fail())
      *p = (unsigned char)n;
  }

}

// --------------------------------------------------------------------------

void FSORB::toMat32F(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }

  const size_t N = descriptors.size();

  mat.create(N, FSORB::L*8, CV_32F);
  float *p = mat.ptr<float>();

  for(size_t i = 0; i < N; ++i)
  {
    const int C = (descriptors[i].first).cols;
    const unsigned char *desc = (descriptors[i].first).ptr<unsigned char>();

    for(int j = 0; j < C; ++j, p += 8)
    {
      p[0] = (desc[j] & (1 << 7) ? 1.f : 0.f);
      p[1] = (desc[j] & (1 << 6) ? 1.f : 0.f);
      p[2] = (desc[j] & (1 << 5) ? 1.f : 0.f);
      p[3] = (desc[j] & (1 << 4) ? 1.f : 0.f);
      p[4] = (desc[j] & (1 << 3) ? 1.f : 0.f);
      p[5] = (desc[j] & (1 << 2) ? 1.f : 0.f);
      p[6] = (desc[j] & (1 << 1) ? 1.f : 0.f);
      p[7] = (desc[j] & (1)      ? 1.f : 0.f);
    }
  }
}

// --------------------------------------------------------------------------

void FSORB::toMat32F(const cv::Mat &descriptors, cv::Mat &mat)
{
  descriptors.convertTo(mat, CV_32F);
}

// --------------------------------------------------------------------------

void FSORB::toMat8U(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  mat.create(descriptors.size(), FSORB::L, CV_8U);

  unsigned char *p = mat.ptr<unsigned char>();

  for(size_t i = 0; i < descriptors.size(); ++i, p += FSORB::L)
  {
    const unsigned char *d = (descriptors[i].first).ptr<unsigned char>();
    std::copy(d, d + FSORB::L, p);
  }

}

// --------------------------------------------------------------------------

bool FSORB::isSemantic()
{
    return true;
}

// --------------------------------------------------------------------------

} // namespace DBoW2

