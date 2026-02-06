// --------------------------------------------------------------
//  Track the parameters and states of CTRNNs as they undergo 
//  Activity-Dependent Homeostatic Plasticity
//  Record pyloric fitness before and after plasticity
//
//  Run this script from the directory of the evolved pyloric circuit
//  which you are perturbing/measuring, and specify the ADHP mechanism
//  you want to test. If you don't want ADHP to be active (just to test
//  pyloricness), then choose the nullADHP.dat file. Only the parameters 
//  indicated by your ADHP mechanism (first line) can be perturbed in the test.

//  Ensure that the ./res.dat file specifies ranges (step doesn't matter)
//  for each parameter that should be explored (lines read in order of plasticpars.dat)
//
//  Test one specific point by setting num_ICs to 1 and the upper and lower bounds
//  of the ranges in res.dat to the values you want
//
// --------------------------------------------------------------

#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"
#include "pyloric.h"

// Simulation parameters
const double TransientDuration = 50;  //Seconds to equilibrate before measuring pyloricness and activating ADHP
double PlasticDuration = 10000; //Seconds with ADHP running before re-measuring pyloricness (set to 0 if just measuring pyl)

const bool trackoutputs = true;
const int trackoutputsinterval = 1; //Track neural outputs for every X trials
const bool trackparams = false;
const int trackparamsinterval = 1; //Track biases for every X trials
const int trackingstepinterval = 2; //make the tracking files smaller by only recording every Xth step (though all steps are integrated)

const int num_ICs = 1; //how many initial points?

//Input Files
char Nfname[] = "./pyloriccircuit.ns";
// char HPfname[] = "./0/bestind.dat";
//null ADHP option
char HPfname[] = "../../nullADHP.dat";
char rangefname[] = "../../res.dat";

//Output Files
char Fitnessesfname[] = "./fit.dat";               //fitness of every point before and after regulation
char ICsfname[] = "./ics.dat";                     //full parameters of every point before and after regulation
char biastrackfname[] = "./parstrack.dat";         //track all plastic parameters throughout the run (if trackparams==true)
char statestrackfname[] = "./statestrack.dat";     //track all three neural output timeseries throughout the run (if trackoutputs==true)

// Nervous system params
const int N = 3;
int	CTRNNVectSize = N*N + 2*N;
int paramboundVectSize = 2*(CTRNNVectSize - N);

void GenPhenMapping(TVector<double> &gen, TVector<double> &phen, TVector<double> &parambounds)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), .1, 2); // Time constants cannot be perturbed or regulated
		k++;
	}
    int param_idx = 1;
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), parambounds(param_idx), parambounds(param_idx+1));
		k++;
        param_idx += 2;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            phen(k) = MapSearchParameter(gen(k), parambounds(param_idx), parambounds(param_idx+1));
            k++;
            param_idx += 2;
        }
	}
}

int main(){
    // Create files to hold data
	ofstream fitnesses;
    fitnesses.open(Fitnessesfname);
    ofstream ICsfile;
    ICsfile.open(ICsfname);
	ofstream biastrack;
    biastrack.open(biastrackfname);
	ofstream statestrack;
	statestrack.open(statestrackfname);

    CTRNN Circuit(3);

    // Set circuit parameters (start with the given pyloric solution)
    ifstream ifs;
    ifs.open(Nfname);
    if (!ifs) {
        cerr << "File not found: " << Nfname << endl;
        exit(EXIT_FAILURE);
    }
    ifs >> Circuit; 
    ifs.close();

    ifstream rangefile;
    rangefile.open(rangefname);
    if (!rangefile) {
        cerr << "File not found: " << rangefname << endl;
        exit(EXIT_FAILURE);
    }

    // Set the proper HP parameters 
    ifstream HPifs;
    HPifs.open(HPfname);
    if (!HPifs) {
        cerr << "File not found: " << HPfname << endl;
        exit(EXIT_FAILURE);
    }
    Circuit.SetHPPhenotype(HPifs,StepSize);
    HPifs.close();

    bool ADHPon = false;
    TVector<double> parambounds(1,paramboundVectSize); //vector to hold all ranges
    if (Circuit.plasticitypars.Sum()>0){ //gets from the ADHP bestind.dat file
        ADHPon = true;
        int bound_idx = 1;
        // Read in specified Ranges
        for (int i=1;i<=Circuit.plasticitypars.UpperBound();i++){
            if (Circuit.plasticitypars(i) == 1){
                int step_throwaway;
                rangefile >> parambounds(bound_idx);
                rangefile >> parambounds(bound_idx+1);
                rangefile >> step_throwaway;
            }
            bound_idx += 2;
        }
    }
    else {PlasticDuration = 0;} //if no ADHP, then forego the plastic period
    rangefile.close();

    // Generate random circuit parameters within the allowed ranges
    TVector<double> genotype(1,CTRNNVectSize);
    TVector<double> phenotype(1,CTRNNVectSize);

    for (int i = 0;i<num_ICs;i++){
        long randomseed = static_cast<long>(time(NULL));
        // long randomseed = 123456789; //if need repeats or direct compare
        RandomState rs(randomseed+pow(i,2));

        for (int j = 1; j <= genotype.Size(); j++)
            {genotype[j] = rs.UniformRandom(-1,1);} //generate random genotype
        
        GenPhenMapping(genotype,phenotype,parambounds); //map into proper ranges (or to specific values)

        //use only the generated parameters that you need
        int k = 1; 
        //check for biases
        for(int j=1; j<=N; j++){
            if (Circuit.plasticitypars[k]==1){
                Circuit.SetNeuronBias(j,phenotype(k+N)); //start after time constants
            }
            k++;
        }

        //check for weights
        for (int j=1; j<=N; j++){
            for (int l=1; l<=N; l++){
                if (Circuit.plasticitypars[k]==1){
                    Circuit.SetConnectionWeight(j,l,phenotype(k+N)); //started after time constants
                }
                k++;
            }
        }

        //prepare circuit for run
        Circuit.RandomizeCircuitOutput(0.5,0.5);
        Circuit.WindowReset();

        // Run for transient without ADHP
        int tstep = 0;
        for(double t=0;t<TransientDuration;t+=StepSize){
            Circuit.EulerStep(StepSize,false);
            if (trackoutputs && (i%trackoutputsinterval==0) && (tstep % trackingstepinterval==0)){
            for (int j = 1; j <= Circuit.size; j++){
                statestrack << Circuit.NeuronOutput(j) << " ";
            }
            statestrack << endl;}
            tstep ++;
        }

        // Record initial parameters
        ICsfile << Circuit.taus << " " << Circuit.biases << " ";
        for(int j = 1; j <= N; j ++){
            for(int k=1;k<=N;k++){
                ICsfile << Circuit.ConnectionWeight(j,k) << " ";
            }
        }
        ICsfile << endl;

        // Run with HP for a time if ADHP is turned on
        for(double t=0;t<PlasticDuration;t+=StepSize){
            if (trackparams && (i%trackparamsinterval==0) && (tstep % trackingstepinterval == 0)){
                for(int j = 1; j<= Circuit.plasticitypars.Sum(); j++){
                    biastrack << Circuit.ArbDParam(j) << " "; //record only the parameters that are changing throughout the run
                }
                biastrack << endl;
            }
			if (trackoutputs && (i%trackoutputsinterval==0) && (tstep % trackingstepinterval==0)){
                for (int j = 1; j <= Circuit.size; j++){
                    statestrack << Circuit.NeuronOutput(j) << " ";
                }
                statestrack << endl;}
            Circuit.EulerStep(StepSize,true);
            tstep ++;
        }

        if (trackparams && (i%trackparamsinterval==0)) {biastrack << endl;}
		if (trackoutputs && (i%trackoutputsinterval==0)) {statestrack << endl;}

        // Record again, after HP
        ICsfile << Circuit.taus << " " << Circuit.biases << " ";
        for(int j = 1; j <= N; j ++){
            for(int k=1;k<=N;k++){
                ICsfile << Circuit.ConnectionWeight(j,k) << " ";
            }
        } 
        ICsfile << endl << endl;

        // Test for Pyloricness (HP remains on if it was on during plastic period)
        double fit = PyloricPerformance(Circuit,true);

        fitnesses << fit << endl << endl;;

        // fitnesses << Circuit.rhos << endl << endl; //proxy for whether HP is satisfied at the end, or whether it just ran into a boundary or is in a limit cycle
    }
    fitnesses.close();
    ICsfile.close();
    biastrack.close();
	statestrack.close();
    return 0;
}