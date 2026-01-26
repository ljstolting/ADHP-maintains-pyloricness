// ************************************************************
// 
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
  // cout << "structor started" << endl;
	size = newsize;
  // cout << "checkpoint test 1" << endl;
	states.SetBounds(1,size);
	states.FillContents(0.0);
	outputs.SetBounds(1,size);
	outputs.FillContents(0.0);
	biases.SetBounds(1,size);
	biases.FillContents(0.0);
  // cout << "checkpoint test 2" << endl;
	gains.SetBounds(1,size);
	gains.FillContents(1.0);
	taus.SetBounds(1,size);
	taus.FillContents(1.0);
  // cout << "checkpoint test 3" << endl;
	Rtaus.SetBounds(1,size);
	Rtaus.FillContents(1.0);
	externalinputs.SetBounds(1,size);
	externalinputs.FillContents(0.0);
	weights.SetBounds(1,size,1,size);
	weights.FillContents(0.0);
	// TempStates.SetBounds(1,size);
	// TempOutputs.SetBounds(1,size);
  stepnum = 0;

  // cout << "basics initialized" << endl;

  // NEW
  rhos.SetBounds(1,size);
  rhos.FillContents(0.0);
  l_boundary.SetBounds(1,size);
  l_boundary.FillContents(0.0);
  // for(int i=1;i<=size;i++){
  //   l_boundary[i] = 0;
  // }
  u_boundary.SetBounds(1,size);
  // for(int i=1;i<=size;i++){
  //   u_boundary[i] = 1;
  // }
  u_boundary.FillContents(1.0);
  tausBiases.SetBounds(1,size);
  RtausBiases.SetBounds(1,size);
  // for(int i=1;i<=size;i++){
  //   tausBiases[i] = 1;
  //   RtausBiases[i] = 1/tausBiases[i];
  // }
  tausBiases.FillContents(1.0);
  RtausBiases.FillContents(1.0);
  tausWeights.SetBounds(1,size,1,size);
  RtausWeights.SetBounds(1,size,1,size);
  // for(int i=1;i<=size;i++){
  //   for(int j=1;j<=size;j++){
  //     tausWeights[i][j] = 1;
  //     RtausWeights[i][j] = 1/(tausWeights[i][j]);
  //   }
  // }
  tausWeights.FillContents(1.0);
  RtausWeights.FillContents(1.0);

  // NEW for AVERAGING
  windowsize.SetBounds(1,size);
  windowsize.FillContents(1); 
  sumoutputs.SetBounds(1,size);
  sumoutputs.FillContents(0);
  avgoutputs.SetBounds(1,size);
  // for(int i=1;i<=size;i++){
  //   avgoutputs[i] = (l_boundary[i]+u_boundary[i])/2; //average of the upper and lower boundaries ensures that initial value keeps HP off
  // }
  avgoutputs.FillContents(0.5);

  minavg.SetBounds(1,size);
  minavg.FillContents(1);
  maxavg.SetBounds(1,size);
  maxavg.FillContents(0);
  // cout << "outputhist about to be created" << endl;
  outputhist.SetBounds(1,windowsize.Sum()); //changing the outputhistory matrix into a big long vector
  // cout << "created:" << outputhist.LowerBound() << " " << outputhist.UpperBound() << endl;
  outputhist.FillContents(0.0);
  outputhiststartidxs.SetBounds(1,size);
  int cumulative = 1;
  for (int neuron = 1; neuron <= size; neuron++){
    outputhiststartidxs(neuron) = cumulative;
    cumulative += windowsize(neuron);
  }

  // NEW for CAPPING
  wr = 16;
  br = 16;

  // Shifted rho default to true
  shiftedrho = true;

  plasticitypars.SetBounds(1,size+(size*size)); //which parameters are under HP's control
  plasticneurons.SetBounds(1,size); //and therefore, which neurons do we need to define a range for 
                                    //*note that the range and sliding window is shared for bias and 
                                    //all incoming weights to a neuron, but each of these parameters 
                                    //may have a different time constant
  plasticitypars.FillContents(0);
  plasticneurons.FillContents(0);
}


// *******
// Control
// *******

// Reset all sliding window utilities and the step counter (crucial if using the same circuit between parameter resets)
void CTRNN::WindowReset(){
  // cout << "Window Resetting" << endl;
  minavg.FillContents(1);
  maxavg.FillContents(0);
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
        if(avgoutputs(i)<minavg(i)){minavg(i)=avgoutputs(i);}; //calc of max and min detected values
        if(avgoutputs(i)>maxavg(i)){maxavg(i)=avgoutputs(i);};
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
        if(avgoutputs(i)<minavg(i)){minavg(i)=avgoutputs(i);}; //calc of max and min detected values
        if(avgoutputs(i)>maxavg(i)){maxavg(i)=avgoutputs(i);};
        // cout << "averages" << avgoutputs << endl;
      }
      // cout << "checkpoint 3" << endl;
    }

    // SHIFTED RHO METHOD(new): Update rho for each neuron
    if (shiftedrho){
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

    // SCALED RHO METHOD(default): Update rho for each neuron.
    else{
      for (int i = 1; i <= size; i++) {
        // cout << l_boundary[i] << " " << u_boundary[i] << endl;
        if (avgoutputs[i] < l_boundary[i]) {
          // cout << l_boundary[i] << endl;
          rhos[i] = (l_boundary[i] - avgoutputs[i])/l_boundary[i];
          // cout << l_boundary[i] << endl << endl;
          // cout << rhos[i] << " " << endl;
        }
        else{
          if (avgoutputs[i] > u_boundary[i]){
            // cout << u_boundary[i] << endl;
            rhos[i] = (u_boundary[i] - avgoutputs[i])/(1.0 - u_boundary[i]);
            // cout << u_boundary[i] << endl << endl;
            // cout << rhos[i] << " " << endl;
          }
          else
          {
            rhos[i] = 0.0; 
          }
        }
        // cout << l_boundary[i] << " " << u_boundary[i] << endl << endl;
      }
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

void CTRNN::PrintMaxMinAvgs(void){
  cout << "Minimum detected:" << minavg << endl;
  cout << "Maximum detected:" << maxavg << endl;
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

// Define HP parameters from a phenotype vector
TVector<double>& CTRNN::SetHPPhenotype(TVector<double>& phenotype, double dt, bool range_encoding){
  // cout << "using phenotype vector" << endl;
  ifstream plasticpars;
  plasticpars.open("../../plasticpars.dat");
  plasticpars >> plasticitypars;
  SetPlasticityPars(plasticitypars);

  int k = 1;
  int phen_counter = 1;
  // Read the bias time constants 
  for(int i = 1; i <= size; i++){
    if(plasticitypars[i] == 1){
        SetNeuronBiasTimeConstant(i,phenotype[phen_counter]);
        // cout << "tc set" << endl;
        phen_counter++;
    } 
    k ++;
  }

  //Weight time constants
  for(int i = 1; i <= size; i++){
    for(int j = 1; j <= size; j++){
      if(plasticitypars[k] == 1){
        SetConnectionWeightTimeConstant(i,j,phenotype[phen_counter]);
        phen_counter++;
      }
      k ++;
    }
  }


  // Read the lower bounds
  for(int i = 1; i<= size; i++){
    if (plasticneurons[i] == 1){
      SetPlasticityLB(i,phenotype[phen_counter]);
      // cout << "lb set" << endl;
      phen_counter++;
    }
  }

  if(range_encoding){
    int num_plastic_neurons = plasticneurons.Sum();
    // Read the ranges and derive the upper bounds
    for(int i = 1; i<= size; i++){
      if (plasticneurons[i] == 1){
        double ub = phenotype[phen_counter-num_plastic_neurons] + phenotype[phen_counter];
        SetPlasticityUB(i,ub); //clipping is built into the set function
        // cout << "ub set" << endl;
        phen_counter ++;
      }
    }
  }

  else
  {  // Read the upper bounds
    for(int i = 1; i<= size; i++){
      if (plasticneurons[i] == 1){
        SetPlasticityUB(i,phenotype[phen_counter]);
        phen_counter++;
      }
    }
  }

  // Read the sliding windows
  for(int i = 1; i<= size; i++){
    if (plasticneurons[i] == 1){
      SetSlidingWindow(i,phenotype[phen_counter],dt);
      // cout << "sw set" << endl;
      phen_counter++;
    }
    k++;
  }

  // IT IS CRUCIAL TO FIX THE SLIDING WINDOW AVERAGING BEFORE EVALUATION
  // Just in case there is not a transient long enough to fill up the history before HP needs to activate
  max_windowsize = windowsize.Max();
  avgoutputs.SetBounds(1,size);
  outputhist.SetBounds(1,windowsize.Sum());
  int cumulative = 1;
  for (int neuron = 1; neuron <= size; neuron++){
    outputhiststartidxs(neuron) = cumulative;
    cumulative += windowsize(neuron);
  }
  WindowReset();
  // cout << outputhist << endl;
	return phenotype;
}

// Define the HP mechanism based on an input file (bestind file from evolution)

istream& CTRNN::SetHPPhenotype(istream& is, double dt, bool range_encoding){
  // cout << "biases from inside" << biases << endl;
  // cout << "using bestindfile" << endl;
  // Read in the parameter vector (specifying which parameters HP is changing)
  for(int i = 1; i <= size + (size*size); i++){
    is >> plasticitypars[i];
  }
  SetPlasticityPars(plasticitypars);
  //NOW ALL CONTAINED WITHIN THAT FUNCTION
  // //determine which neurons need to have ranges and windows
  // for(int i=1;i<=size;i++){
  //   //check biases
  //   plasticneurons[i] = plasticitypars[i];
  // }

  // // Determine if only weights or biases are changing - speeds up the code
  // int k = 1;
  // adaptbiases = false;
  // adaptweights = false;
  // for (int i=1;i<=size;i++){
  //   if (plasticitypars[k] == 1) {adaptbiases = true;}
  //   k ++;
  // }
  
  // for (int i=1;i<=(size*size);i++){
  //   if (plasticitypars[k] == 1) {adaptweights = true;}
  //   k ++;
  // }

  // for(int i=1;i<=size;i++){
  //   //check incoming weights if not already flagged
  //   if (plasticitypars[i] == 0){
  //     for (int j=0;j<=(plasticitypars.UpperBound()-i-size);j+=size){
  //       if (plasticitypars[size+i+j] == 1){
  //         plasticneurons[i] = 1;
  //         break;
  //       }
  //     }
  //   }
  // }

  // cout << "plasticneurons set by HPphenotype:" << plasticneurons << endl;

   // pass by the genotype until you get to the first value of the phenotype, which will be a time constant >= 100
  double testvar = 0;
  int varloops = 0;
  while (testvar < 100){
    is >> testvar;
    varloops ++;
    if (varloops > 100){
      cerr << "check HP phenotype file" << endl;
      exit(EXIT_FAILURE);
    }
    // cout << testvar << " ";
  }
  // cout << endl;
  bool use_flag = 0;
  int k = 1;
  // Read the bias time constants
  double btau;
  for(int i = 1; i <= size; i++){
    if(plasticitypars[k] == 1){
      if (!use_flag){
        SetNeuronBiasTimeConstant(i,testvar);
        use_flag = 1;
      }
      else{
        is >> btau;
        // cout << btau << " ";
        SetNeuronBiasTimeConstant(i,btau);
      }
    } 
    k ++;
  }
  
  // Read the weight time constants 
  double wtau;
  for(int i = 1; i <= size; i++){
    for(int j = 1; j <= size; j++){
      if(plasticitypars[k] == 1){
        is >> wtau;
        // cout << wtau << " ";
        SetConnectionWeightTimeConstant(i,j,wtau);
      }
      k ++;
    }
  }
  
  // Read the lower bounds
  TVector<double> lbs(1,size);
  for(int i = 1; i<= size; i++){
    if (plasticneurons[i] == 1){
      is >> lbs[i];
      // cout << lbs[i] << " ";
      SetPlasticityLB(i,lbs[i]);
    }
  }

  if(range_encoding){
    // Read the ranges and derive the upper bounds
    double range;
    for(int i = 1; i<= size; i++){
      if (plasticneurons[i] == 1){
        is >> range;
        // cout << range << " ";
        double ub = lbs[i] + range;
        SetPlasticityUB(i,ub); //clipping is built into the set function
      }
    }
  }

  else
  {  // Read the upper bounds
    double ub;
    for(int i = 1; i<= size; i++){
      if (plasticneurons[i] == 1){
        is >> ub;
        SetPlasticityUB(i,ub);
      }
    }
  }

  // Read the sliding windows
  double sw;
  for(int i = 1; i<= size; i++){
    if (plasticneurons[i] == 1){
      is >> sw;
      // cout << sw << " ";
      SetSlidingWindow(i,sw,dt);
    }
  }

  double fitness;
  is >> fitness; // just trying something
  // cout << endl << "all HP params set" << endl;
  // IT IS CRUCIAL TO FIX THE SLIDING WINDOW AVERAGING BEFORE EVALUATION
  // Just in case there is not a transient long enough to fill up the history before HP needs to activate
  max_windowsize = windowsize.Max();
  // cout << "Max window size:" << max_windowsize << endl;
  avgoutputs.SetBounds(1,size);
  outputhist.SetBounds(1,windowsize.Sum());
  int cumulative = 1;
  for (int neuron = 1; neuron <= size; neuron++){
    outputhiststartidxs(neuron) = cumulative;
    cumulative += windowsize(neuron);
  }
  WindowReset();
  // cout << "SetHPPhenotype complete" << endl;
	return is;
}

void CTRNN::WriteHPGenome(ostream& os){

  // os << setprecision(32);
  // write the bias time constants
  for (int i = 1; i<=num_pars_changed; i++){
    os << NeuronBiasTimeConstant(i) << " ";
  }
	os << endl << endl;

  // write the lower bounds
  for (int i = 1; i<=num_pars_changed; i++){
    os << PlasticityLB(i) << " ";
  }
  os << endl;

  // write the upper bounds
  for (int i = 1; i<=num_pars_changed; i++){
    os << PlasticityUB(i) << " ";
  }
  os << endl << endl;

  // write the sliding windows
  for (int i = 1; i<=num_pars_changed; i++){
    os << SlidingWindow(i) << " ";
  }

	return;
}

void CTRNN::SetHPPhenotypebestind(istream &is, double dt, bool range_encoding){
  for(int i = 1; i <= size + (size*size); i++){
    is >> plasticitypars[i];
  }
  SetPlasticityPars(plasticitypars);
  cout << num_pars_changed;
  TVector<double> gen(1,num_pars_changed*4); 
  TVector<double> phen(1,gen.UpperBound());
  double fit;

  is >> gen;
  is >> phen;
  cout << phen << endl;
  is >> fit;

  // cout << gen << endl << phen << endl;

  // cout << trial << endl<< gen << endl << phen << endl;
  SetHPPhenotype(phen,dt,range_encoding);
}


// ****************
// Input and Output
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
  // cout << size << endl;
  c.size = size;
	// Read the time constants
	for (int i = 1; i <= size; i++) {
		is >> c.taus[i];
    // cout << c.taus[i] << " ";
		c.Rtaus[i] = 1/c.taus[i];
	}
  // cout << endl;
	// Read the biases
	for (int i = 1; i <= size; i++){
		is >> c.biases[i];
  }
  // cout << c.biases << endl;
	// Read the gains
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
  // IT IS CRUCIAL TO FIX THE SLIDING WINDOW AVERAGING BEFORE EVALUATION
  c.max_windowsize = c.windowsize.Max();
  // cout << "checkpoint 1" << endl;
	c.avgoutputs.SetBounds(1,c.size);
	for(int i=1;i<=c.size;i++){
	  c.avgoutputs[i] = (c.l_boundary[i]+c.u_boundary[i])/2; //average of the upper and lower boundaries ensures that initial value keeps HP off
	}
  // cout << "checkpoint 2" << endl;
	c.outputhist.SetBounds(1,c.windowsize.Sum());
  c.outputhist.FillContents(0.0);
  int cumulative = 1;
  for (int neuron = 1; neuron <= c.size; neuron++){
    c.outputhiststartidxs(neuron) = cumulative;
    cumulative += c.windowsize(neuron);
  }  //some number that would never be taken on by the neurons
  // cout << "checkpoint 3" << endl;
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
  // cout << "taus written" << endl;
	// Write the biases
	for (int i = 1; i <= c.size; i++){
		phen[k] = c.biases[i];
    k ++;
  }
  // cout << "biases written" << endl;

	// Write the weights
	for (int i = 1; i <= c.size; i++) {
		for (int j = 1; j <= c.size; j++){
			phen[k] = c.weights[i][j];
      k ++;
    }
	}
  // cout << "weights written" << endl;

	// Return the phenotype
	return phen;
}
