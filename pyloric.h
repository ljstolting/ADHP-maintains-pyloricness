// ---------------------------------------------------------------
//  Family of functions for evaluating CTRNNs on their pyloricness
//
//  Lindsay Stolting 4/6/24
//
// TO DO:
// - remove legacy mode
// - ensure step size always inherited from here
// - check makefile layouts to make sure compiles automatically
// - check whether pragma once works on windows
// - copy and clean main files
// ---------------------------------------------------------------
#ifndef pyloric_h
#define pyloric_h

#include "CTRNN.h"
#include "VectorMatrix.h"
#include <iostream>
#include <cmath>

using namespace std;

// Task params
const bool ADHPtest = true;            //is HP on during pyloric testing (shouldn't matter if platicity time constants are slow enough, but seems to matter in select cases
const double TestDuration = 100;     //maximum number of seconds allowed to locate 3 cycles of the rhythm 
const int TestSteps = TestDuration/StepSize; 

// Detection params
const double burstthreshold = .5;    //threshold that must be crossed for detecting bursts
const double tolerance = 3*StepSize; //tolerance for detecting low-dimensional double periodicity

// Evaluation params (adjust the fitness function)
const double scaling_factor = 0.05;   //how much is awarded for each binary criteria met (changes importance relative to timing award)
const bool timing_award = true;	      //award extra points for timing (true) or cap at reaching all 6 criteria (false)?

// pyloric evaluation functions
void BurstTimesfromOutputHist(TMatrix<double> &OutputHistory, TVector<double> &features, bool debug = false);
double PyloricFitFromFeatures(TVector<double> &FeatureVect, bool debug = false);
double PyloricPerformance(CTRNN &Agent, bool debug = false);
double PyloricPerformance(CTRNN &Agent, ofstream &trajfile, ofstream &burstfile);

// utility functions for generating initial point grids
void converttobase(int N,int resolution,TVector<int> &converted);
void PointCombos(TMatrix<int> &answer,int resolution);
void PointGrid(TMatrix<double> &points, TVector<double> &parvals);

#endif
