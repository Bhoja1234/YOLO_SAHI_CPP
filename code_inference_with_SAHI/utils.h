#pragma once
#include <filesystem>
#include <fstream>
#include <random>

#include "global.h"
#include "RunOnnx.h"
#include "SAHI.h"

//void Detector(DCSP_CORE*& p);
//void Segment(DCSP_CORE*& p);

void DetectTest(bool use_sahi);
void SegmentTest();

void Check_GPU();