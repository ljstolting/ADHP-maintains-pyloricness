// ************************************************************
// CTRNN class implementation with ADHP
// ************************************************************

#include "CTRNN.h"
#include "random.h"
#include "VectorMatrix.h"
#include <stdlib.h>

// A fast sigmoid implementation using a table w/ linear interpolation
#ifdef FAST_SIGMOID
int SigTableInitFlag = 0;
double SigTab[SigTabSize];

void InitSigmoidTable(void)
{
  if (!SigTableInitFlag) {
    double DeltaX = SigTabRange/(SigTabSize-1);
    for (int i = 0; i <= SigTabSize-1; i++)
      SigTab[i] = sigma(i * DeltaX);
    SigTableInitFlag = 1;
  }
}

double fastsigmoid(double x)
{
  if (x >= SigTabRange) return 1.0;
  if (x < 0) return 1.0 - fastsigmoid(-x);
  double id;
  double frac = modf(x*(SigTabSize-1)/SigTabRange, &id);
  int i = (int)id;
  double y1 = SigTab[i], y2 = SigTab[i+1];

  return y1 + (y2 - y1) * frac;
}
#endif

void getNumNeuronsPlastic(TVector<int>& plastic_neurons, TVector<int>& plasticpars,int N){
  //for general use outside of the class
  plastic_neurons.FillContents(0);
  for (int i=plasticpars.LowerBound();i<=plasticpars.UpperBound();i++){
    if (plasticpars(i) == 1){
      plastic_neurons((i - 1) % N + 1) = 1;
    }
  }
  return;
}

// ****************************
// Constructors and Destructors
// ****************************

// The constructor
CTRNN::CTRNN(int newsize)
{
	SetCircuitSize(newsize);
#ifdef FAST_SIGMOID
  InitSigmoidTable();
#endif
}

// The destructor

CTRNN::~CTRNN()
{
	SetCircuitSize(0);
}


// *********
// Utilities
// *********

// Resize a circuit.

void CTRNN::SetCircuitSize(int newsize)
{
  //Simulation step
  stepnum = 0;

  //CTRNN states
	size = newsize;
	states.SetBounds(1,size);
	states.FillContents(0.0);
	outputs.SetBounds(1,size);
	outputs.FillContents(0.0);
	externalinputs.SetBounds(1,size);
	externalinputs.FillContents(0.0);

  //CTRNN parameters
	weights.SetBounds(1,size,1,size);
	weights.FillContents(0.0);
  biases.SetBounds(1,size);
	biases.FillContents(0.0);
	gains.SetBounds(1,size);
	gains.FillContents(1.0);
	taus.SetBounds(1,size);
	taus.FillContents(1.0);
	Rtaus.SetBounds(1,size);
	Rtaus.FillContents(1.0);

  //ADHP metaparameters
  l_boundary.SetBounds(1,size);
  l_boundary.FillContents(0.0);
  u_boundary.SetBounds(1,size);
  u_boundary.FillContents(1.0);
  tausBiases.SetBounds(1,size);
  tausBiases.FillContents(1.0);
  RtausBiases.SetBounds(1,size);
  RtausBiases.FillContents(1.0);
  tausWeights.SetBounds(1,size*size,1,size*size);
  tausWeights.FillContents(1.0);
  RtausWeights.SetBounds(1,size*size,1,size*size);
  RtausWeights.FillContents(1.0);
  windowsize.SetBounds(1,size);
  windowsize.FillContents(1); 

  //ADHP internal states
  rhos.SetBounds(1,size);
  rhos.FillContents(0.0);
  sumoutputs.SetBounds(1,size);
  sumoutputs.FillContents(0);
  avgoutputs.SetBounds(1,size);
  avgoutputs.FillContents(0.5);
  mindetected.SetBounds(1,size);
  mindetected.FillContents(1);
  maxdetected.SetBounds(1,size);
  maxdetected.FillContents(0);

  // Default Boundaries that ADHP can't cross
  wr = 16; //weights
  br = 16; //and biases

  plasticitypars.SetBounds(1,size+(size*size)); //which parameters are under HP's control
  plasticneurons.SetBounds(1,size); //and therefore, which neurons do we need to define a range for? *

  //*note that currently, the range and sliding window is shared for bias 
  //and all incoming weights to a neuron, but each of these parameters 
  //may have a different time constant. other overlaps could be tried

  //Automatically get the plasticity pars from the ./plasticpars.dat file
  char plasticparsfname[] = "./plasticpars.dat"; //may have to change filepath if running from different directory
  ifstream plasticparsfile(plasticparsfname); 
  if (!plasticparsfile.is_open()) {
    cerr << "Could not open file: " << plasticparsfname << endl;
    exit(1);
  }
  plasticparsfile >> plasticitypars;

  SetPlasticityPars(plasticitypars); //also sets plastic neurons and adaptflags
  WindowInitialize();
  WindowReset();
}


// *******
// Control
// *******

void CTRNN::WindowInitialize(void){
  outputhiststartidxs.SetBounds(1,size);
  max_windowsize = windowsize.Max();
  outputhist.SetBounds(1,windowsize.Sum());
  int cumulative = 1;
  // TO DO: should be per parameter, not per neuron
  for (int neuron = 1; neuron <= size; neuron++){
    outputhiststartidxs(neuron) = cumulative;
    cumulative += windowsize(neuron);
  }
}

// Reset all sliding window utilities and the step counter (crucial if using the same circuit between parameter resets)
void CTRNN::WindowReset(){
  // cout << "Window Resetting" << endl;
  mindetected.FillContents(1);
  maxdetected.FillContents(0);
  sumoutputs.FillContents(0);
  outputhist.FillContents(0.0);
  for(int i=1;i<=size;i++){
    avgoutputs[i] = (l_boundary[i]+u_boundary[i])/2; //average of the upper and lower boundaries ensures that initial value keeps HP off
  }
  stepnum = 0;
}


// Randomize the states or outputs of a circuit.

void CTRNN::RandomizeCircuitState(double lb, double ub)
{
	for (int i = 1; i <= size; i++){
      SetNeuronState(i, UniformRandom(lb, ub));
      SetNeuronOutput(i, sigmoid(gains[i] * (states[i] + biases[i])));
  }
  // reset averaging and sliding window utilities
  WindowReset();
}

void CTRNN::RandomizeCircuitState(double lb, double ub, RandomState &rs)
{
	for (int i = 1; i <= size; i++){
    SetNeuronState(i, rs.UniformRandom(lb, ub));
    SetNeuronOutput(i, sigmoid(gains[i] * (states[i] + biases[i])));
  }
  // reset averaging and sliding window utilities
  WindowReset();
}

void CTRNN::RandomizeCircuitOutput(double lb, double ub)
{
	for (int i = 1; i <= size; i++){
      SetNeuronOutput(i, UniformRandom(lb, ub));
      SetNeuronState(i, (InverseSigmoid(outputs[i])/gains[i])-biases[i]);
  }
  // reset averaging and sliding window utilities
  WindowReset();
}

void CTRNN::RandomizeCircuitOutput(double lb, double ub, RandomState &rs)
{
	for (int i = 1; i <= size; i++){
    SetNeuronOutput(i, rs.UniformRandom(lb, ub));
    SetNeuronState(i, (InverseSigmoid(outputs[i])/gains[i])-biases[i]);
  }
  // reset averaging and sliding window utilities
  WindowReset();
}



// Way to check if all the elements of the output array are now valid CTRNN outputs
// bool checkoutputhist(double array[], int size)
// {
//   for (int i = 0; i < size; i++)
//   {
//       if(array[i] < 0)
//           return false; // return false at the first found

//   }
//   return true; //all elements checked
// }

// Update the averages and rhos of a neuron
void CTRNN::RhoCalc(void){
  // Keep track of the running average of the outputs for some predetermined window of time.
    // 1. Window should always stay updated no matter whether adapting or not (faster so not expensive)
    // 2. Take average for each neuron (unless its sliding window has not yet passed; in that case leave average in between ub and lb to turn HP off)
    // cout << "rhocalc called" << endl;
    for (int i = 1; i <= size; i++){
      // cout << stepnum << " " << windowsize[i] << endl;
      int outputhistindex = outputhiststartidxs(i) + ((stepnum) % windowsize(i));
      // cout << outputhiststartidxs << " " << i << " " << outputhistindex << endl;
      if(stepnum < windowsize[i]){
        outputhist(outputhistindex) = NeuronOutput(i);
        // cout << outputhist << endl;
      }
      // cout << "checkpoint 1" << endl;
      if(stepnum == windowsize[i]){ //do initial add-up
        for (int k = 0; k < windowsize[i]; k++){  
          sumoutputs[i] += outputhist(outputhiststartidxs(i)+k);
        }
        // cout << sumoutputs(i) << endl;
        avgoutputs[i] = sumoutputs[i]/windowsize[i];
        if(avgoutputs(i)<mindetected(i)){mindetected(i)=avgoutputs(i);}; //calc of max and min detected values
        if(avgoutputs(i)>maxdetected(i)){maxdetected(i)=avgoutputs(i);};
        // cout << "averages" << avgoutputs << endl;
      }
      // cout << "checkpoint 2" << endl;
      if(stepnum > windowsize[i]){ //do truncated add-up
        //subtract oldest value
        sumoutputs(i) -= outputhist(outputhistindex);
        //add new value
        sumoutputs(i) += NeuronOutput(i);
        // replace oldest value with new one
        outputhist(outputhistindex) = NeuronOutput(i);
        avgoutputs(i) = sumoutputs(i)/windowsize(i);
        if(avgoutputs(i)<mindetected(i)){mindetected(i)=avgoutputs(i);}; //calc of max and min detected values
        if(avgoutputs(i)>maxdetected(i)){maxdetected(i)=avgoutputs(i);};
        // cout << "averages" << avgoutputs << endl;
      }
      // cout << "checkpoint 3" << endl;
    }

    //Update rho for each neuron
    for (int i = 1; i <= size; i++) {
      // cout << l_boundary[i] << " " << u_boundary[i] << endl;
      if (avgoutputs[i] < l_boundary[i]) {
        rhos[i] = -avgoutputs[i]+l_boundary[i];
      }
      else{
        if (avgoutputs[i] > u_boundary[i]){
          rhos[i] = -avgoutputs[i]+u_boundary[i];
        }
        else
        {
          rhos[i] = 0.0; 
        }
      }
      // cout << l_boundary[i] << " " << u_boundary[i] << endl << endl;
    }
  
}

// Integrate a circuit one step using Euler integration.

void CTRNN::EulerStep(double stepsize, bool adaptpars)
{
  // Update the state of all neurons.
  for (int i = 1; i <= size; i++) {
    double input = externalinputs[i];
    for (int j = 1; j <= size; j++) {
      input += weights[j][i] * outputs[j];
    }
    states[i] += stepsize * Rtaus[i] * (input - states[i]);
    outputs[i] = sigmoid(gains[i] * (states[i] + biases[i]));
  } 

  if (adaptpars == true) 
    {
      RhoCalc();
      // if(stepnum<10){cout << "rhos:" << rhos << endl << "sumoutputs: << sumoutputs << endl;}
      stepnum ++;
    
      // NEW: Update Biases
    if(adaptbiases==true){
    //  // cout << "biaschangeflag" << endl;
      for (int i = 1; i <= size; i++){
        if (plasticitypars[i]==1){
          biases[i] += stepsize * RtausBiases[i] * rhos[i];
          if (biases[i] > br){
              biases[i] = br;
          }
          else{
              if (biases[i] < -br){
                  biases[i] = -br;
              }
          }
        }
      } 
    }
    // NEW: Update Weights
    if(adaptweights==true)
    { 
      int k = size;
      // cout << "weightchangeflag" << endl;
      for (int i = 1; i <= size; i++) 
      {
        for (int j = 1; j <= size; j++)
        {
          k ++;
          if(plasticitypars[k] == 1){
            weights[i][j] += stepsize * RtausWeights[i][j] * rhos[j] * fabs(weights[i][j]);

            if (weights[i][j] > wr)
            {
                weights[i][j] = wr;
            }
            else
            {
                if (weights[i][j] < -wr)
                {
                    weights[i][j] = -wr;
                }
            }
          }
        }
      }
    }
  }
}

void CTRNN::EulerStepAvgsnoHP(double stepsize)
// Keeps track of the maxmin averages detected, but does not actually change the circuit parameters
{
  // Update the state of all neurons.
  for (int i = 1; i <= size; i++) {
    double input = externalinputs[i];
    for (int j = 1; j <= size; j++)
      input += weights[j][i] * outputs[j];
    states[i] += stepsize * Rtaus[i] * (input - states[i]);
    outputs[i] = sigmoid(gains[i] * (states[i] + biases[i]));
  }
  RhoCalc();
  stepnum ++;
}



// Set the biases of the CTRNN to their center-crossing values

void CTRNN::SetCenterCrossing(void)
{
    double InputWeights, ThetaStar;

    for (int i = 1; i <= CircuitSize(); i++) {
        // Sum the input weights to this neuron
        InputWeights = 0;
        for (int j = 1; j <= CircuitSize(); j++)
            InputWeights += ConnectionWeight(j, i);
        // Compute the corresponding ThetaStar
        ThetaStar = -InputWeights/2;
        SetNeuronBias(i, ThetaStar);
    }
}
// ***********************
// ADHP reading/writing
// ***********************

// Define ADHP parameters from a phenotype vector
TVector<double>& CTRNN::SetHPPhenotype(TVector<double>& phenotype, double dt){

  //*plasticity pars should already have been set in initialization of CTRNN

  int phen_counter = 1;
  // Read the bias time constants 
  for(int i = 1; i <= size; i++){
    if(plasticitypars[phen_counter] == 1){
        SetNeuronBiasTimeConstant(i,phenotype[phen_counter]);
        phen_counter++;
    } 
  }

  //Weight time constants
  for(int i = 1; i <= size; i++){
    for(int j = 1; j <= size; j++){
      if(plasticitypars[phen_counter] == 1){
        SetConnectionWeightTimeConstant(i,j,phenotype[phen_counter]);
        phen_counter++;
      }
    }
  }

  // Read the lower bounds
  for(int i = 1; i<= size; i++){
    if (plasticneurons[i] == 1){
      SetPlasticityLB(i,phenotype[phen_counter]);
      phen_counter++;
    }
  }

  int num_plastic_neurons = plasticneurons.Sum();

  // Read the ranges and derive the upper bounds
  for(int i = 1; i<= size; i++){
    if (plasticneurons[i] == 1){
      double ub = PlasticityLB(i) + phenotype[phen_counter];
      SetPlasticityUB(i,ub); //clipping is built into the set function
      phen_counter ++;
    }
  }


  // Read the sliding windows
  for(int i = 1; i<= size; i++){
    if (plasticneurons[i] == 1){
      SetSlidingWindow(i,phenotype[phen_counter],dt);
      phen_counter++;
    }
  }

  // IT IS CRUCIAL TO FIX THE SLIDING WINDOW AVERAGING BEFORE EVALUATION
  // Just in case there is not a transient long enough to fill up the history before HP needs to activate
  WindowInitialize();
  WindowReset();

	return phenotype;
}

// Define the HP mechanism based on an input file (bestind file from evolution)

istream& CTRNN::SetHPPhenotype(istream& is, double dt){
  // Read in the parameter vector (specifying which parameters HP is changing), as in this mode it's allowed to be different from the one initialized in the constructor
  for(int i = 1; i <= size + (size*size); i++){
    is >> plasticitypars[i];
  }
  SetPlasticityPars(plasticitypars);

  TVector<double> gen(1,num_pars_changed*4); 
  TVector<double> phen(1,gen.UpperBound());
  double fit;

  is >> gen;
  is >> phen;
  is >> fit;

  SetHPPhenotype(phen,dt);

	return is;
}

void CTRNN::WriteHPGenome(ostream& os){

  // os << setprecision(32);

  // write the bias time constants
  for (int i = 1; i<=size; i++){
    if (plasticitypars[i]==1){
      os << NeuronBiasTimeConstant(i) << " ";
    }
  }
  for (int i = size+1; i <= (size*size)+size; i++){
    if (plasticitypars[i]==1){
      int from = floor((i - size - 1) / size) + 1;
      int to = ((i - size - 1) % size) + 1;
      os << ConnectionWeightTimeConstant(from,to) << " ";
    }
  }
	os << endl << endl;

  // write the lower bounds
  for (int i = 1; i<=num_neurons_plastic; i++){
    os << PlasticityLB(i) << " ";
  }
  os << endl;

  // write the upper bounds
  for (int i = 1; i<=num_neurons_plastic; i++){
    os << PlasticityUB(i) << " ";
  }
  os << endl << endl;

  // write the sliding windows
  for (int i = 1; i<=num_neurons_plastic; i++){
    os << SlidingWindow(i) << " ";
  }

	return;
}


// ****************
// CTRNN Reading/Writing
// ****************

#include <iomanip>

ostream& operator<<(ostream& os, CTRNN& c)
{//NOT UPDATED TO READ OUT HP PARAMETERS
	// Set the precision
	os << setprecision(8);
	// Write the size
	os << c.size << endl << endl;
	// Write the time constants
	for (int i = 1; i <= c.size; i++)
		os << c.taus[i] << " ";
	os << endl << endl;
	// Write the biases
	for (int i = 1; i <= c.size; i++)
		os << c.biases[i] << " ";
	os << endl << endl;
	// Write the gains
	for (int i = 1; i <= c.size; i++)
		os << c.gains[i] << " ";
	os << endl << endl;
	// Write the weights
	for (int i = 1; i <= c.size; i++) {
		for (int j = 1; j <= c.size; j++)
			os << c.weights[i][j] << " ";
		os << endl;
	}
	// Return the ostream
	return os;
}

istream& operator>>(istream& is, CTRNN& c)
{//NOT UPDATED TO READ IN HP PARAMETERS
	// Read the size
	int size;
	is >> size;
  c.size = size;

	// Read the time constants
	for (int i = 1; i <= size; i++) {
		is >> c.taus[i];
    // cout << c.taus[i] << " ";
		c.Rtaus[i] = 1/c.taus[i];
	}

	// Read the biases
	for (int i = 1; i <= size; i++){
		is >> c.biases[i];
  }

	// Read the gains (not part of this project, but included for consistency)
	for (int i = 1; i <= size; i++){
		is >> c.gains[i];
  }

	// Read the weights
	for (int i = 1; i <= size; i++){
		for (int j = 1; j <= size; j++){
			is >> c.weights[i][j];
    }
  }
	// Return the istream
	return is;
}

TVector<double>& operator>>(TVector<double>& phen , CTRNN& c)
{//NOT UPDATED TO READ IN HP PARAMETERS
  int k = 1;
	// Read the time constants
	for (int i = 1; i <= c.size; i++) {
		c.taus[i] = phen[k];
		c.Rtaus[i] = 1/c.taus[i];
    k ++;
	}
	// Read the biases
	for (int i = 1; i <= c.size; i++){
		c.biases[i] = phen[k];
    k ++;
  }
	// Read the weights
	for (int i = 1; i <= c.size; i++){
		for (int j = 1; j <= c.size; j++){
			c.weights[i][j] = phen[k];
      k ++;
    }
  }
  // Should be null, but just to be safe, FIX THE SLIDING WINDOW AVERAGING BEFORE EVALUATION
  c.WindowInitialize();
  c.WindowReset();

	// Return the phenotype
	return phen;
}

TVector<double>& operator<<(TVector<double>& phen, CTRNN& c)
{//NOT UPDATED TO READ OUT HP PARAMETERS -- still CTRNN parameters
  int k = 1;
	// Write the time constants
	for (int i = 1; i <= c.size; i++){
		phen[k] = c.taus[i];
    k++;
  }
	// Write the biases
	for (int i = 1; i <= c.size; i++){
		phen[k] = c.biases[i];
    k ++;
  }

	// Write the weights
	for (int i = 1; i <= c.size; i++) {
		for (int j = 1; j <= c.size; j++){
			phen[k] = c.weights[i][j];
      k ++;
    }
	}

	// Return the phenotype
	return phen;
}
