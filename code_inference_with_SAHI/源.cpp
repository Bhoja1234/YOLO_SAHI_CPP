#include <iostream>
#include <iomanip>

#include "utils.h"

int main()
{
    // ���GPU
    Check_GPU();

    bool use_sahi = true;
    DetectTest(use_sahi);

    return 0;
}
