#include <iostream>
#include <cmath>


extern float __enzyme_autodiff(void*, ...);
int enzyme_const, enzyme_dup, enzyme_out;

static inline float clamp(float u, float a, float b) { return std::min(std::max(a, u), b); }

static inline float FrDielectricPBRT(float cosThetaI, float etaI, float etaT) 
{
  cosThetaI = clamp(cosThetaI, -1.0f, 1.0f);
  // Potentially swap indices of refraction
  bool entering = cosThetaI > 0.0f;
  if (!entering) 
  {
    const float tmp = etaI;
    etaI = etaT;
    etaT = tmp;
    cosThetaI = std::abs(cosThetaI);
  }

  // Compute _cosThetaT_ using Snell's law
  float sinThetaI = std::sqrt(std::max(0.0f, 1.0f - cosThetaI * cosThetaI));
  float sinThetaT = etaI / etaT * sinThetaI;

  // Handle total internal reflection
  if (sinThetaT >= 1.0f) 
    return 1.0f;

  const float cosThetaT = std::sqrt(std::max(0.0f, 1.0f - sinThetaT * sinThetaT));
  const float Rparl     = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
  const float Rperp     = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
  return 0.5f*(Rparl * Rparl + Rperp * Rperp);
}

 struct float3
  {
    inline float3() : x(0), y(0), z(0) {}
    inline float3(float a_x, float a_y, float a_z) : x(a_x), y(a_y), z(a_z) {}
    inline explicit float3(float a_val) : x(a_val), y(a_val), z(a_val) {}
    inline explicit float3(const float a[3]) : x(a[0]), y(a[1]), z(a[2]) {}
    
    inline float& operator[](int i)       { return M[i]; }
    inline float  operator[](int i) const { return M[i]; }

    union
    {
      struct { float x, y, z; };
      float M[3];
    };
  };

  static inline float3 operator+(const float3 a, const float3 b) { return float3{a.x + b.x, a.y + b.y, a.z + b.z}; }
  static inline float3 operator-(const float3 a, const float3 b) { return float3{a.x - b.x, a.y - b.y, a.z - b.z}; }
  static inline float3 operator*(const float3 a, const float3 b) { return float3{a.x * b.x, a.y * b.y, a.z * b.z}; }
  static inline float3 operator/(const float3 a, const float3 b) { return float3{a.x / b.x, a.y / b.y, a.z / b.z}; }

  static inline float3 operator * (const float3 a, float b) { return float3{a.x * b, a.y * b, a.z * b}; }
  static inline float3 operator / (const float3 a, float b) { return float3{a.x / b, a.y / b, a.z / b}; }
  static inline float3 operator * (float a, const float3 b) { return float3{a * b.x, a * b.y, a * b.z}; }
  static inline float3 operator / (float a, const float3 b) { return float3{a / b.x, a / b.y, a / b.z}; }

  static inline float3 operator + (const float3 a, float b) { return float3{a.x + b, a.y + b, a.z + b}; }
  static inline float3 operator - (const float3 a, float b) { return float3{a.x - b, a.y - b, a.z - b}; }
  static inline float3 operator + (float a, const float3 b) { return float3{a + b.x, a + b.y, a + b.z}; }
  static inline float3 operator - (float a, const float3 b) { return float3{a - b.x, a - b.y, a - b.z}; }

  static inline float3& operator *= (float3& a, const float3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z;  return a; }
  static inline float3& operator /= (float3& a, const float3 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z;  return a; }
  static inline float3& operator *= (float3& a, float b) { a.x *= b; a.y *= b; a.z *= b;  return a; }
  static inline float3& operator /= (float3& a, float b) { a.x /= b; a.y /= b; a.z /= b;  return a; }

  static inline float3& operator += (float3& a, const float3 b) { a.x += b.x; a.y += b.y; a.z += b.z;  return a; }
  static inline float3& operator -= (float3& a, const float3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z;  return a; }
  static inline float3& operator += (float3& a, float b) { a.x += b; a.y += b; a.z += b;  return a; }
  static inline float3& operator -= (float3& a, float b) { a.x -= b; a.y -= b; a.z -= b;  return a; }


  static inline float3 min  (const float3 a, const float3 b) { return float3{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
  static inline float3 max  (const float3 a, const float3 b) { return float3{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
  static inline float3 clamp(const float3 u, const float3 a, const float3 b) { return float3{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y), clamp(u.z, a.z, b.z)}; }
  static inline float3 clamp(const float3 u, float a, float b) { return float3{clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b)}; }

  static inline float3 abs (const float3 a) { return float3{std::abs(a.x), std::abs(a.y), std::abs(a.z)}; } 

  static inline float3 lerp(const float3 a, const float3 b, float t) { return a + t * (b - a); }
  static inline float3 mix (const float3 a, const float3 b, float t) { return a + t * (b - a); }
  static inline float3 floor(const float3 a)                { return float3{std::floor(a.x), std::floor(a.y), std::floor(a.z)}; }
  static inline float3 ceil(const float3 a)                 { return float3{std::ceil(a.x), std::ceil(a.y), std::ceil(a.z)}; }
  static inline float3 rcp (const float3 a)                 { return float3{1.0f/a.x, 1.0f/a.y, 1.0f/a.z}; }
  static inline float3 mod (const float3 x, const float3 y) { return x - y * floor(x/y); }
  static inline float3 fract(const float3 x)                { return x - floor(x); }
  static inline float3 sqrt(const float3 a)                 { return float3{std::sqrt(a.x), std::sqrt(a.y), std::sqrt(a.z)}; }
  static inline float3 inversesqrt(const float3 a)          { return 1.0f/sqrt(a); }
  
  static inline  float dot(const float3 a, const float3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }

  static inline  float length(const float3 a) { return std::sqrt(dot(a,a)); }
  static inline  float3 normalize(const float3 a) { float lenInv = float(1)/length(a); return a*lenInv; }


static inline float GGX_Distribution(const float cosThetaNH, const float alpha)
{
  const float alpha2 = alpha * alpha;
  const float NH_sqr = clamp(cosThetaNH * cosThetaNH, 0.0f, 1.0f);
  const float den    = NH_sqr * alpha2 + (1.0f - NH_sqr);
  return alpha2 / std::max((float)(M_PI) * den * den, 1e-6f);
}

static inline float GGX_GeomShadMask(const float cosThetaN, const float alpha)
{
  // Height - Correlated G.
  //const float tanNV      = sqrt(1.0f - cosThetaN * cosThetaN) / cosThetaN;
  //const float a          = 1.0f / (alpha * tanNV);
  //const float lambda     = (-1.0f + sqrt(1.0f + 1.0f / (a*a))) / 2.0f;
  //const float G          = 1.0f / (1.0f + lambda);

  // Optimized and equal to the commented-out formulas on top.
  const float cosTheta_sqr = clamp(cosThetaN * cosThetaN, 0.0f, 1.0f);
  const float tan2         = (1.0f - cosTheta_sqr) / std::max(cosTheta_sqr, 1e-6f);
  const float GP           = 2.0f / (1.0f + std::sqrt(1.0f + alpha * alpha * tan2));
  return GP;
}


static inline float ggxEvalBSDF(const float3 l, const float3 v, const float3 n, const float roughness)
{
  if(std::abs(dot(l, n)) < 1e-5f)
    return 0.0f; 
 
  const float dotNV = dot(n, v);  
  const float dotNL = dot(n, l);
  if (dotNV < 1e-6f || dotNL < 1e-6f)
    return 0.0f; 

  const float  roughSqr = roughness * roughness;
  const float3 h    = normalize(v + l); // half vector.
  const float dotNH = dot(n, h);
  const float D     = GGX_Distribution(dotNH, roughSqr);
  const float G     = GGX_GeomShadMask(dotNV, roughSqr)*GGX_GeomShadMask(dotNL, roughSqr);      

  return (D * G / std::max(4.0f * dotNV * dotNL, 1e-6f));  // Pass single-scattering
}


float testFunc(const float* in_data)
{
   const float arg1 = std::abs(in_data[0]);
   const float arg2 = std::abs(in_data[1]);
   const float arg3 = std::abs(in_data[2]);
   
   const float ggxVal = ggxEvalBSDF(float3(arg1, arg2, arg3), 
                                     float3(arg1-arg2, arg2-arg3, arg3-arg1), 
                                     float3(arg1+arg2, arg2+arg3, arg3+arg1), arg1);

   const float tmp1 = FrDielectricPBRT(arg1, ggxVal, arg3);
   const float tmp2 = FrDielectricPBRT(tmp2, arg2+tmp2, arg1)*ggxVal;

   return std::sqrt(ggxVal);
}

int main(int argc, const char** argv)
{
  float data[3] = {0.5f,0.5f,1.0f};
  float grad[3] = {0,0,0};

  __enzyme_autodiff((void*)testFunc, 
                    enzyme_dup, data, grad);

  std::cout << "grad = " << grad[0] << ", " << grad[1] << ", " << grad[2] << std::endl;
  return 0;
}