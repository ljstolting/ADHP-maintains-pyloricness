// ***********************************************************
// CTRNN class with ADHP
// ************************************************************
#ifndef CTRNN_h
#define CTRNN_h

const double StepSize = 0.05;        //THE MASTER LOCATION TO CHANGE STEP SIZE FOR ALL SIMULATIONS

// Uncomment the following line for table-based fast sigmoid w/ linear interpolation
//#define FAST_SIGMOID

#include "VectorMatrix.h"
#include "random.h"
#include <iostream>
#include <math.h>

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

void getNumNeuronsPlastic(TVector<int>& plastic_neurons, TVector<int>& plasticpars,int N);


// The CTRNN class declaration
class CTRNN {
    public:
        // The constructor
        CTRNN(int size);
        // The destructor
        ~CTRNN();

        // Accessors
        int CircuitSize(void) {return size;};
        void SetCircuitSize(int newsize); //called in constructor/destructor
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

        //"ArbDParam": arbitrary dimension parameter, way to map between list of ADHP controlled 
        // parameters and their identity within the circuit
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
      
        double NeuronRho(int i) {return rhos[i];};
        void SetNeuronRho(int i, double value) {rhos[i] = value;};
        double PlasticityLB(int i) {return l_boundary[i];};
        void SetPlasticityLB(int i, double value) {
          if (value <0){
            value = 0;
          }
          if (value > 1){
            value = 1;
          }
          l_boundary(i) = value;
          //clipping copied here for safety
        }
        double PlasticityUB(int i) {return u_boundary[i];};
        void SetPlasticityUB(int i, double value) {
          if (value < 0){
            value = 0;
          }
          //must be used *after* PlasticityLB has been correctly set for proper protections
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
        void SetSlidingWindow(int i, double windsize, double dt) {windowsize(i)=1+int(windsize/dt);};
        void DetectionReset(void){
          mindetected.FillContents(1.0);
          maxdetected.FillContents(0.0);
        }
        void PrintMaxMinDetected(void){
          cout << "Minimum detected:" << mindetected << endl;
          cout << "Maximum detected:" << maxdetected << endl;
        }
        void SetCenterCrossing(void);
        istream& SetHPPhenotype(istream& is, double dt);
        TVector<double>& SetHPPhenotype(TVector<double>& phenotype, double dt);
        void WriteHPGenome(ostream& os);

        // Input and output
        friend ostream& operator<<(ostream& os, CTRNN& c);
        friend TVector<double>& operator<<(TVector<double>& phen, CTRNN& c);
        friend istream& operator>>(istream& is, CTRNN& c);
        friend TVector<double>& operator>>(TVector<double>& phen, CTRNN& c);

        // Control
        void WindowInitialize(void);
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
        int max_windowsize, num_pars_changed, num_neurons_plastic;
        bool adaptbiases, adaptweights;
        TVector<double> states, outputs, biases, gains, taus, Rtaus, externalinputs;
        TVector<double> rhos, tausBiases, RtausBiases, l_boundary, u_boundary, mindetected, maxdetected; // NEW
        TVector<double> avgoutputs, sumoutputs, outputhist; // Note outputhist is a vector instead of a matrix, where all neuron histories are concatenated
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
            if (plasticitypars[i] == 1)
              {
                adaptbiases = true;
                plasticneurons[i] = 1;
              }
          }
          //check weights
          for(int i=size+1;i<=plasticitypars.UpperBound();i++){
            if (plasticitypars[i] == 1) 
              {
                adaptweights = true;
                plasticneurons[((i - 1) % size) + 1] = 1;
              }
          }
          num_pars_changed = plasticpars.Sum();
          num_neurons_plastic = plasticneurons.Sum();
        };
};
#endif