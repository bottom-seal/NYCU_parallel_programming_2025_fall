#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  //  ANS: because the code will write to some memory that is not allocated to the array in last write back, violate the memory.
  //  to address this issue, we should modify maskAll, making it mask exceeding positions in last iteration.
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // Fix: the memory violation issue
    int remaining = N - i;
    int width = (remaining < VECTOR_WIDTH) ? remaining : VECTOR_WIDTH;

    // All ones
    //maskAll = _pp_init_ones();

    // Fix: only enables legal positions
    maskAll = _pp_init_ones(width);

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {
    // fix: maskAll on maskisNegative left poses beyond bound stay at 0, doing not on it will accidently make it 1, which access illegal memory.
    // we need to do and with maskall, to ensure out of bound part is kept as 0(masked)
    maskIsNotNegative = _pp_mask_and(maskIsNotNegative, maskAll);

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  //vector x, for float array x in serial(values)
  __pp_vec_float x;
  //vector y, for int array y in serial(exponents)
  __pp_vec_int y;
  //vector result holds value to write back
  __pp_vec_float result;
  //vector zero, seiral compared y[i] with 0 to see if we should just output 1 for x[i]
  __pp_vec_int zero = _pp_vset_int(0);
  //vector one's, y vector need to sub by 1 before entering while loop
  __pp_vec_int one = _pp_vset_int(1); 
  //vector for limit of 9.999999f
  __pp_vec_float limit = _pp_vset_float(9.999999f);
  //masks to use
  __pp_mask maskAll, maskOutput1, maskOutputElse, maskClamp;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // first we calculate enabled width to avoid memory violation.
    int remaining = N - i;
    int width = (remaining < VECTOR_WIDTH) ? remaining : VECTOR_WIDTH;
    // intialize maskAll, we only need to do operation on legal positions
    maskAll = _pp_init_ones(width);
    //initialize other mask
    maskOutput1 = _pp_init_ones(0);
    maskOutputElse = _pp_init_ones(0);
    maskClamp = _pp_init_ones(0);
    //loads from x array
    _pp_vload_float(x, values + i, maskAll);
    //loads from y array
    _pp_vload_int(y, exponents + i, maskAll);
    //if y[i] == 0, maskOutput1 will have 1 in the pos if maskAll allows
    _pp_veq_int(maskOutput1, y, zero, maskAll);// if (y == 0)
    //sets x[i] to 1 where y[i] is 0
    _pp_vset_float(result, 1, maskOutput1);
    //sets mask to know what pos we should work on, this mask excludes already written pos and where maskAll doesn't allow.
    maskOutputElse = _pp_mask_not(maskOutput1); // else
    //applying and with maskAll to ensure no illegal memory access
    maskOutputElse = _pp_mask_and(maskOutputElse, maskAll);
    //this part implements while loop
    _pp_vsub_int(y, y, one, maskOutputElse);// int count = y - 1;
    //copies x to result
    _pp_vmove_float(result, x, maskOutputElse);
    while (_pp_cntbits(maskOutputElse) != 0) { // when all bits are masked ( == 0), operation is done.
      //now y holds count
      //!!! not sure if having result and reference mask being same one cause bug
      //compares count with 0, if greater mask = 1, will do operation
      _pp_vgt_int(maskOutputElse, y, zero, maskAll); //while (count > 0)
      _pp_vmult_float(result, result, x, maskOutputElse);// result *= x
      _pp_vsub_int(y, y, one, maskOutputElse);// count--
    }
    //if (result > 9.999999f)
    _pp_vgt_float(maskClamp, result, limit, maskAll);
    //result = 9.999999f;
    _pp_vset_float(result, 9.999999f, maskClamp);
    //write back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  //vector to hold result
  __pp_vec_float result;
  //we need mask that enable all channel for loading
  __pp_mask maskAll;
  maskAll = _pp_init_ones();
  float total = 0;
  int steps = (int)log2(VECTOR_WIDTH) - 1; // concluded this from experiment.
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  // from experience i found that we can get a vector with each element = sum, after doing (hadd + interleave) for (n - 1) times, where WIDTH = 2^n, then do a hadd.
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
     _pp_vload_float(result, values + i, maskAll);
    //doing the loop once would have half of orignal vector represent added up number (half of vec wrt last iteration added up = value for whole vector in last iteration)
    for(int j = 0; j < steps; j++)
    {
      _pp_hadd_float(result, result);
      _pp_interleave_float(result, result);
    }
    //need to do hadd one more time (concluded from experiment)
    _pp_hadd_float(result, result);
    //the vec is filled with added up sum for 
    total += result.value[0];
  }

  return total;
 

  
}
