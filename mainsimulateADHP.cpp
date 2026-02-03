// --------------------------------------------------------------
//  Track the parameters of many CTRNNs as they undergo Homeostatic Plasticity
// --------------------------------------------------------------
#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"
#include "pyloric.h"

//#define PRINTOFILE

// Task params
const double TransientDuration = 50; //Seconds with HP off
const double PlasticDuration = 10000; //Seconds with HP running (or if HP is off, how long to record the trajectory)
// const double StepSize = 0.025;

// Nervous system params
const int N = 3;
const bool shiftedrho_tf = true;
int	CTRNNVectSize = N*N + 2*N;

// Pyloric Detection params
// const double burstthreshold = .5; //threshold that must be crossed for detecting bursts
// const double tolerance = .01; //for detecting double periodicity

// Sampling Parameters

const double TMIN = .1;
const double TMAX = 2;
// note that the bounds for the second neuron's bias will not be used if the HP file does not indicate that that bias is being regulated
// SINGLE POINT MODE
// const double BRlb1 = 10;   
// const double BRub1 = BRlb1;
// const double BRlb2 = 0;
// const double BRub2 = BRlb2;
// const double BRlb3 = 10;
// const double BRub3 = BRlb3;
// -0.4 -14.3 2.58

// RANDOM POINTS MODE 
const double BRlb1 = -16;   
const double BRub1 = 16;
const double BRlb2 = -16;
const double BRub2 = 16;
const double BRlb3 = -16;
const double BRub3 = 16;

const double WR = 20;
const int num_ICs = 25;

// Mode
const bool random_mode = false; //randomize in other dimensions besides HP dimensions 
const bool taus_set = false; //in random mode, do we want the taus to be variable

//Filenames
char Nfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/pyloriccircuit.ns";
char HPfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/6/bestind.dat";
// char HPfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/3D3/HP3Dfor15.dat";
// char HPfname[] = "./bestindtest.dat";
// char HPfname[] = "./Convenient HP Mechanisms/nullHP3D.dat";
// char HPfname[] = "./Specifically Evolved HP mechanisms/Pete/2D/33/bestind.dat";
// char Fitnessesfname[] = "./Convenient HP Mechanisms/Petefitbad.dat";
char Fitnessesfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/6/fit.dat";
// char Fitnessesfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/Other Plane Trajectories/fit.dat";
// char Fitnessesfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/testfit.dat";
// char ICsfname[] = "./Convenient HP Mechanisms/Peteicsbad.dat";
// char ICsfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/Other Plane Trajectories/ics.dat";
char ICsfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/6/ics.dat";
// char ICsfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/testics.dat";
// char biastrackfname[] = "./Convenient HP Mechanisms/Petebiastrackbad.dat";
char biastrackfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/6/biastrack.dat";
// char biastrackfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/testbiastrack.dat";
// char biastrackfname[] = "Specifically Evolved HP mechanisms/Every Circuit/1/Other Plane Trajectories/biastrack.dat";
// char statestrackfname[] = "./Convenient HP Mechanisms/Petestatestrackbad.dat";
// char statestrackfname[] = "./teststatestrack.dat";
char statestrackfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/6/statestrack.dat";
// char statestrackfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/teststatestrack.dat";
// char statestrackfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/Other Plane Trajectories/evolved.dat";

const bool trackstates = false;
const int trackstatesinterval = 1; //Track neural outputs for every X trials
const bool trackparams = true;
const int trackparamsinterval = 1; //Track biases for every X trials
const int trackingstepinterval = 2; //make the tracking files smaller by only recording every Xth step (though all steps are integrated)

void GenPhenMapping(TVector<double> &gen, TVector<double> &phen, TVector<double> &biasbounds)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), biasbounds(i), biasbounds(i+N));
		k++;
	}
    // cout << phen(k);
	// Weights
	for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            phen(k) = MapSearchParameter(gen(k), -WR, WR);
            k++;
        }
	}
}

int main(){
    TVector<double> biasbounds(1,2*N);
    biasbounds(1) = BRlb1;
    biasbounds(2) = BRlb2;
    biasbounds(3) = BRlb3;
    biasbounds(4) = BRub1;
    biasbounds(5) = BRub2;
    biasbounds(6) = BRub3;

    // cout << StepSize << endl;
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
    // cout << Circuit.l_boundary << " " << Circuit.u_boundary << endl;
    // cout << Circuit.br;

    // Set circuit parameters (start with the given pyloric solution)
    ifstream ifs;
    ifs.open(Nfname);
    if (!ifs) {
        cerr << "File not found: " << Nfname << endl;
        exit(EXIT_FAILURE);
    }
    ifs >> Circuit; 
    Circuit.ShiftedRho(shiftedrho_tf);

    // Set the proper HP parameters 
    ifstream HPifs;
    HPifs.open(HPfname);
    if (!HPifs) {
        cerr << "File not found: " << HPfname << endl;
        exit(EXIT_FAILURE);
    }
    // if(HPfileisbestind){
    //     Circuit.SetHPPhenotypebestind(HPifs,StepSize,true);
    // }
    Circuit.SetHPPhenotype(HPifs,StepSize,true);

    // cout << Circuit.PlasticityLB(1) << " " << Circuit.PlasticityLB(2) << " " << Circuit.PlasticityLB(3) << " " << Circuit.PlasticityUB(1) << " " << Circuit.PlasticityUB(2) << " " << Circuit.PlasticityUB(3) << endl;
    // cout << Circuit.SlidingWindow(1) << endl;

    // Set of random circuit parameters to pull from
    TVector<double> genotype(1,CTRNNVectSize);
    TVector<double> phenotype(1,CTRNNVectSize);

    for (int i = 0;i<num_ICs;i++){
        long randomseed = static_cast<long>(time(NULL));
        // long randomseed = 6942069420; //if need repeats or direct compare
        RandomState rs(randomseed+pow(i,2));
        for (int j = 1; j <= genotype.Size(); j++)
            {genotype[j] = rs.UniformRandom(-1,1);}
        
        GenPhenMapping(genotype,phenotype,biasbounds);

        // FULLY RANDOM MODE (well everything but the time constants)
        if(random_mode){
            int k = 1;
            if(!taus_set){
                for(int j=1;j<=N;j++){
                    Circuit.SetNeuronTimeConstant(j,phenotype(k));
                    k++;
                }
            }
            k = N+1; //start after time constants
            for(int j=1; j<=N; j++){
                Circuit.SetNeuronBias(j,phenotype(k));
                // cout << "set a bias" << endl;
                k++;
            }

            for (int j=1; j<=N; j++){
                for (int l=1; l<=N; l++){
                    Circuit.SetConnectionWeight(j,l,phenotype(k));
                    // cout << "set a weight" << endl; 
                    k ++;   
                }
            }
        }

        // RANDOM IN HP DIMENSIONS MODE (SLICE MODE)
        else{
            int k = 1; 
            for(int j=1; j<=N; j++){
                //check for biases
                if (Circuit.plasticitypars[k]==1){
                    Circuit.SetNeuronBias(j,phenotype(k+N)); //start after time constants
                    // cout << phenotype(k+N);
                    // Circuit.SetNeuronBias(j,-10); //if want a specific value
                    // cout << "set a bias" << endl;
                }
                k++;
            }

            //check for weights
            for (int j=1; j<=N; j++){
                for (int l=1; l<=N; l++){
                    if (Circuit.plasticitypars[k]==1){
                        Circuit.SetConnectionWeight(j,l,phenotype(k+N)); //started after time constants
                        // cout << "set a weight" << endl;
                    }
                    k++;
                }
            }
        }
        // cout << Circuit.biases << endl << Circuit.weights << endl;

        Circuit.RandomizeCircuitOutput(0.5,0.5);
        Circuit.WindowReset();

        // Run for transient without HP
        for(double t=0;t<TransientDuration;t+=StepSize){
            Circuit.EulerStep(StepSize,false);
        }

        // Record all parameters
        ICsfile << Circuit.taus << " " << Circuit.biases << " ";
        for(int j = 1; j <= N; j ++)
        {
            for(int k=1;k<=N;k++)
            {
                ICsfile << Circuit.ConnectionWeight(j,k) << " ";
            }
        }
        ICsfile << endl;

        // Run with HP for a time
        int tstep = 0;
        for(double t=0;t<PlasticDuration;t+=StepSize){
            if (trackparams && (i%trackparamsinterval==0) && (tstep % trackingstepinterval == 0)){
                for(int j = 1; j<= Circuit.plasticitypars.Sum(); j++){
                    biastrack << Circuit.ArbDParam(j) << " "; //record only the parameters that are changing throughout the run
                }
                biastrack << endl;
            }
			if (trackstates && (i%trackstatesinterval==0) && (tstep%trackingstepinterval==0)){
                for (int j = 1; j <= Circuit.size; j++){
                    statestrack << Circuit.NeuronOutput(j) << " ";
                }
                statestrack << endl;}
            Circuit.EulerStep(StepSize,true);
            tstep ++;
        }
        //Adding an HP off period for testing purposes 
        for(double t=0;t<TransientDuration;t+=StepSize){
            if (trackparams && (i%trackparamsinterval==0) && (tstep % trackingstepinterval == 0)){
                for(int j = 1; j<= Circuit.plasticitypars.Sum(); j++){
                    biastrack << Circuit.ArbDParam(j) << " "; //record only the parameters that are changing throughout the run
                }
                biastrack << endl;
            }
			if (trackstates && (i%trackstatesinterval==0) && (tstep%trackingstepinterval==0)){
                for (int j = 1; j <= Circuit.size; j++){
                    statestrack << Circuit.NeuronOutput(j) << " ";
                }
                statestrack << endl;}
            Circuit.EulerStep(StepSize,false);
            tstep ++;
        }

        if (trackparams && (i%trackparamsinterval==0)) {biastrack << endl;}
		if (trackstates && (i%trackstatesinterval==0)) {statestrack << endl;}

        // Record again, after HP
        ICsfile << Circuit.taus << " " << Circuit.biases << " ";
        for(int j = 1; j <= N; j ++){
            for(int k=1;k<=N;k++){
                ICsfile << Circuit.ConnectionWeight(j,k) << " ";
            }
        } 
        ICsfile << endl << endl;

        // Test for Pyloricness with or without HP (HPtest in pyloric.h file)
        // double fit = PyloricPerformance(Circuit,biastrack,statestrack);
        double fit = PyloricPerformance(Circuit,TransientDuration);

        fitnesses << fit << endl;
        cout << fit << endl;
        // if(fit > .3){cout << "pyloric found" << endl;}
        fitnesses << Circuit.rhos << endl << endl; //proxy for whether HP is satisfied at the end, or whether it just ran into a boundary or is in a limit cycle

        if (trackstates){statestrack << Circuit.outputs<< " " << endl;}
    }
    fitnesses.close();
    ICsfile.close();
    biastrack.close();
	statestrack.close();
    return 0;
}