#include <iostream>

extern float __enzyme_autodiff(void*, ...);
int enzyme_const, enzyme_dup, enzyme_out;

float testFunc(const float* in_data)
{
   const float arg1 = in_data[0];
   const float arg2 = in_data[1];
   const float arg3 = in_data[2];

   return (arg1+arg2)*arg3;
}

int main(int argc, const char** argv)
{
  float data[3] = {1,2,3};
  float grad[3] = {0,0,0};

  __enzyme_autodiff((void*)testFunc, 
                    enzyme_dup, data, grad);

  std::cout << "grad = " << grad[0] << ", " << grad[1] << ", " << grad[2] << std::endl;
  return 0;
}