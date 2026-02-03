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
const double TestDuration = 100;     //maximum number of seconds allowed to locate 3 cycles of the rhythm
const bool HPequilibrate = false;    //is HP on during the transient/equilibration period? 
const bool HPtest = true;            //is HP on during test (shouldn't matter if platicity time constants are slow enough, *****but seems to be mattering in select cases****)
const double StepSize = 0.05;        //THE MASTER LOCATION TO CHANGE STEP SIZE FOR ALL SIMULATIONS
const int TestSteps = TestDuration/StepSize; 

// Detection params
const double burstthreshold = .5;    //threshold that must be crossed for detecting bursts
const double tolerance = 3*StepSize; //for detecting double periodicity

// Evaluation params (adjust the fitness function)
const double scaling_factor = 0.05;   //how much is awarded for each binary criteria met (changes importance relative to timing award)
const bool timing_award = true;	      //award extra points for timing (true) or cap at reaching all 6 criteria (false)?
const bool legacy = false;            //use the old set of timing criteria from the Prinz paper using PD end as anchor (LPstart<PYstart, LPend<PYend, PDend<LPstart)-true
									  //or the new one using LPstart as anchor (LP starts in silence, PYstart<LPend, LPend<PYend)-false

void BurstTimesfromOutputHist(TMatrix<double> &OutputHistory, TVector<double> &features);
double PyloricFitFromFeatures(TVector<double> &FeatureVect);
double PyloricPerformance(CTRNN &Agent);
double PyloricPerformance(CTRNN &Agent, double TransientDur);
double PyloricPerformance(CTRNN &Agent, ofstream &trajfile, ofstream &burstfile);
void converttobase(int N,int resolution,TVector<int> &converted);
void PointCombos(TMatrix<int> &answer,int resolution);
void PointGrid(TMatrix<double> &points, TVector<double> &parvals);

#endif
