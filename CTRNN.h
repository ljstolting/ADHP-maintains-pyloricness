// ***********************************************************
// A class for continuous-time recurrent neural networks
//
// RDB
//  8/94 Created
//  12/98 Optimized integration
//  1/08 Added table-based fast sigmoid w/ linear interpolation
// ************************************************************

// Uncomment the following line for table-based fast sigmoid w/ linear interpolation
//#define FAST_SIGMOID

#include "VectorMatrix.h"
#include "random.h"
#include <iostream>
#include <math.h>

#pragma once


// The sigmoid function

#ifdef FAST_SIGMOID
const int SigTabSize = 400;
const double SigTabRange = 15.0;

double fastsigmoid(double x);
#endif

inline double sigma(double x)
{
  return 1/(1 + exp(-x));
}

inline double sigmoid(double x)
{
#ifndef FAST_SIGMOID
  return sigma(x);
#else
  return fastsigmoid(x);
#endif
}


// The inverse sigmoid function

inline double InverseSigmoid(double y)
{
  return log(y/(1-y));
}


// The CTRNN class declaration

class CTRNN {
    public:
        // The constructor
        CTRNN(int size);
        // The destructor
        ~CTRNN();

        // Accessors
        int CircuitSize(void) {return size;};
        void SetCircuitSize(int newsize);
        double NeuronState(int i) {return states[i];};
        double &NeuronStateReference(int i) {return states[i];};
        void SetNeuronState(int i, double value)
            {states[i] = value;outputs[i] = sigmoid(gains[i]*(states[i] + biases[i]));};
        double NeuronOutput(int i) {return outputs[i];};
        double &NeuronOutputReference(int i) {return outputs[i];};
        void SetNeuronOutput(int i, double value)
            {outputs[i] = value; states[i] = InverseSigmoid(value)/gains[i] - biases[i];};
        double NeuronBias(int i) {return biases[i];};
        void SetNeuronBias(int i, double value) {biases[i] = value;};
        double NeuronGain(int i) {return gains[i];};
        void SetNeuronGain(int i, double value) {gains[i] = value;};
        double NeuronTimeConstant(int i) {return taus[i];};
        void SetNeuronTimeConstant(int i, double value) {taus[i] = value;Rtaus[i] = 1/value;};
        double NeuronExternalInput(int i) {return externalinputs[i];};
        double &NeuronExternalInputReference(int i) {return externalinputs[i];};
        void SetNeuronExternalInput(int i, double value) {externalinputs[i] = value;};
        double ConnectionWeight(int from, int to) {return weights[from][to];};
        void SetConnectionWeight(int from, int to, double value) {weights[from][to] = value;};
        // --NEW FOR GENERAL HP MECHANISM: the arbitrary dimension parameters which are subject
        //                                 to change are indexed from 1 through num in the order
        //                                 to which they are referred in the plasticpars vector
        //                                 (biases then weights in from-to order)
        void SetArbDParam(int i, double value) {
          int par_index = 0;
          int k = 0;
          while (k < i){
            par_index++;
            if (plasticitypars(par_index) == 1){
              k++;
            }
          }
          if (par_index <= size){
            SetNeuronBias(par_index,value);
          }
          else{
            par_index --; //treat as if were zero indexing
            int from = floor(par_index/size); //comes out in one indexing because of the biases
            int to = par_index % size;
            to ++; //change to one indexing
            SetConnectionWeight(from,to,value);
          }
        }
        double ArbDParam(int i) {
          int par_index = 0;
          int k = 0;
          while (k < i){
            par_index++;
            if (plasticitypars(par_index) == 1){
              k++;
            }
          }
          if (par_index <= size){
            return NeuronBias(par_index);
          }
          else{
            par_index --; //treat as if were zero indexing
            int from = floor(par_index/size); //comes out in one indexing because of the biases
            int to = par_index % size;
            to ++; //change to one indexing
            return ConnectionWeight(from,to);
          }
        }
      
        // -- NEW
        double NeuronRho(int i) {return rhos[i];};
        void SetNeuronRho(int i, double value) {rhos[i] = value;};
        void ShiftedRho(bool tf){shiftedrho = tf;};
        double PlasticityLB(int i) {return l_boundary[i];};
        void SetPlasticityLB(int i, double value) {
          if (value <0){
            value = 0;
          }
          if (value > 1){
            value = 1;
          }
          l_boundary(i) = value;
          } //putting clipping here just to be safe
        double PlasticityUB(int i) {return u_boundary[i];};
        void SetPlasticityUB(int i, double value) {
          //must always be used after SetPlasticityLB
          if (value < 0){
            value = 0;
          }
          else if (value < l_boundary[i]){
            value = l_boundary[i];
          }
          else if (value > 1){
            value = 1;
          }
          u_boundary(i) = value;
          } //putting clipping here just to be safe
        double NeuronBiasTimeConstant(int i) {return tausBiases[i];};
        void SetNeuronBiasTimeConstant(int i, double value) {tausBiases[i] = value; RtausBiases[i] = 1/value;};
        double ConnectionWeightTimeConstant(int from, int to) {return tausWeights[from][to];};
        void SetConnectionWeightTimeConstant(int from, int to, double value) {tausWeights[from][to] = value; RtausWeights[from][to] = 1/value;};
        int SlidingWindow(int i) {return windowsize[i];};
        // Built in protections against changing step sizes -- entered SW is always time-based
        void SetSlidingWindow(int i, double windsize, double dt) 
        {
          windowsize(i)=1+int(windsize/dt);
        };
        void SetMaxavg(int i, double a) {maxavg[i] = a;};
        void SetMinavg(int i, double a) {minavg[i] = a;};
        // --
        void LesionNeuron(int n)
        {
            for (int i = 1; i<= size; i++) {
                SetConnectionWeight(i,n,0);
                SetConnectionWeight(n,i,0);
            }
        }
        void SetCenterCrossing(void);
        istream& SetHPPhenotype(istream& is, double dt, bool range_encoding);
        TVector<double>& SetHPPhenotype(TVector<double>& phenotype, double dt, bool range_encoding);
        void WriteHPGenome(ostream& os);
        void SetHPPhenotypebestind(istream& is, double dt, bool range_encoding);
        void PrintMaxMinAvgs(void);

        // Input and output
        friend ostream& operator<<(ostream& os, CTRNN& c);
        friend TVector<double>& operator<<(TVector<double>& phen, CTRNN& c);
        friend istream& operator>>(istream& is, CTRNN& c);
        friend TVector<double>& operator>>(TVector<double>& phen, CTRNN& c);

        // Control
        void WindowReset(void);
        void RandomizeCircuitState(double lb, double ub);
        void RandomizeCircuitState(double lb, double ub, RandomState &rs);
        void RandomizeCircuitOutput(double lb, double ub);
        void RandomizeCircuitOutput(double lb, double ub, RandomState &rs);
        void RhoCalc(void);
        void EulerStep(double stepsize, bool adaptpars);
        void EulerStepAvgsnoHP(double stepsize);
        // void RK4Step(double stepsize);

        int size, stepnum;
        TVector<int> windowsize, plasticitypars, plasticneurons, outputhiststartidxs; // NEW for AVERAGING
        double wr, br; // NEWER for CAPPING
        int max_windowsize, num_pars_changed;
        bool adaptbiases, adaptweights, shiftedrho;
        TVector<double> states, outputs, biases, gains, taus, Rtaus, externalinputs;
        TVector<double> rhos, tausBiases, RtausBiases, l_boundary, u_boundary, minavg, maxavg; // NEW
        TVector<double> avgoutputs, sumoutputs, outputhist; // NEW for AVERAGING, change outputhist into a vector
        TMatrix<double> weights;
        TMatrix<double> tausWeights, RtausWeights; // NEW
        void SetPlasticityPars(TVector<int>& plasticpars){
          plasticitypars=plasticpars;
          // determine if only weights or only biases are changed
          adaptbiases = false;
          adaptweights = false;
          for(int i=1;i<=size;i++){
            //check biases
            plasticneurons[i] = plasticitypars[i];
            if (plasticitypars[i] == 1){adaptbiases = true;}
          }

          for(int i=size+1;i<=plasticitypars.UpperBound();i++){
            if (plasticitypars[i] == 1) {adaptweights = true;}
          }
          //determine which neurons need to have ranges and windows
          for(int i=1;i<=size;i++){
            //check incoming weights if not already flagged
            if (plasticitypars[i] == 0){
              for (int j=0;j<=(plasticitypars.UpperBound()-i-size);j+=size){
                if (plasticitypars[size+i+j] == 1){
                  plasticneurons[i] = 1;
                  break;
                }
              }
            }
          }
          num_pars_changed = plasticpars.Sum();
        };
        // TVector<double> TempStates,TempOutputs;
};
