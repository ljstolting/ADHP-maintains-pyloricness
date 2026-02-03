// --------------------------------------------------------------
//  Evolve an ADHP mechanism to recover pyloric function best from 
//  a set of evenly spaced starting conditions
// --------------------------------------------------------------

#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"
#include "pyloric.h"
#include "VectorMatrix.h"
#include <stdio.h>
#include <cstring>
#include <sys/stat.h> 

// Task params
const double TransientDuration = 50; //seconds without HP
const double PlasticDuration1 = 5000; //seconds allowing HP to act before first test
const double PlasticDuration2 = 5000; //seconds to wait before testing again, to make sure not relying on precise timing

// EA params
const int POPSIZE = 25;
const int GENS = 1;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;
const double pyloric_boost = 25; //fitness to add for each instance of full pyloric recovery
const bool seed_center_crossing = false; //initialize the population with center-crossing networks?

// Nervous system params
const int N = 3;
const bool shiftedrho_tf = true;

//which file specifies circuit genome?
// 1: different individual in each parent folder - ideal for parallel evolutions
//-----------------------------------------------------------
// const char circuitfname[] = "../pyloriccircuit.ns";
//-----------------------------------------------------------

// 2: one specific individual for this run 
//-----------------------------------------------------------
const char circuitfname[] = "./Timing Requirements/92/pyloriccircuit.ns";
//-----------------------------------------------------------

// ADHP params
//which file specifies plasticity access?
const char plasticparsfname[] = "./plasticpars.dat";

// Plasticity metaparameter ranges
const double SWMIN = 0;		// Window Size min (in seconds)
const double SWMAX = 10;		// Window Size max (in seconds)
const double LBMIN = 0;			// Target Range Lower Bound min
const double LBMAX = 1;			// Target Range Lower Bound max
const double RANGEMIN = 0; 		// Target Range Width min
const double RANGEMAX = 1; 		// Target Range Width max
const double TMIN = 100.0;		// parameter Time Constant min
const double TMAX = 200.0;		// parameter Time Constant max

// ------------------------------------
// Genotype-Phenotype Mapping Function
// Transform genotype on range [0,1] to specified phenotype ranges
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen, int num_pars_changed, int num_neurons_plastic)
{
	// inherit the number of parameters controlled by ADHP from the length of the given genotype vector

	int k = 1; //carry over indexer
	// Parameter Time-constants
	for (int i = 1; i <= num_pars_changed; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Lower Bounds
	for (int i = 1; i <= num_neurons_plastic; i++) {
		phen(k) = MapSearchParameter(gen(k), LBMIN, LBMAX);
		k++;
	}
	// Range Width -- upper bound is clipped when necessary
	for (int i = 1; i <= num_neurons_plastic; i++) {
		phen(k) = MapSearchParameter(gen(k), RANGEMIN, RANGEMAX);
		k++;
	}
    // Sliding Window -- duration gets rounded to the nearest stepsize in the SetSlidingWindow function
    for (int i = 1; i <= num_neurons_plastic; i++) {
		phen(k) = MapSearchParameter(gen(k), SWMIN, SWMAX);
		k++;
	}
}

double HPPerformance(CTRNN &Agent, TMatrix<double> &ptlist, double pyloric_boost){
	// Evaluate the performance of the circuit across all points in ptlist, with some pyloric bonus
	double fitness = 0;
	int num_points = ptlist.RowUpperBound();

	for (int i = 1; i <= num_points; i ++){
		for (int b=1;b<=Agent.num_pars_changed;b++){
			Agent.SetArbDParam(b,ptlist(i,b)); 
		}
		// Clear any sliding window history
		Agent.WindowReset();

		// Initialize the outputs at 0.5 for all neurons in the circuit
		Agent.RandomizeCircuitOutput(.5,.5);

		// Run the circuit for an initial transient; ADHP is off and fitness is not evaluated
		for (double t = StepSize; t <= TransientDuration; t += StepSize) {
			Agent.EulerStep(StepSize,false);
		}

		// Run the circuit for a period of time with ADHP so the paramters can change
		for (double t = StepSize; t<= PlasticDuration1; t+= StepSize){
			Agent.EulerStep(StepSize,true); 
		}

		// Calculate the Pyloric Fitness
		double fit = PyloricPerformance(Agent);

		// Award bonus if fully pyloric
		if (fit >= .3){fit = fit+pyloric_boost;}

		// Add to cumulative fitness
		fitness += fit;

		// repeat process after another duration of time to ensure solution is stable
		for (double t = StepSize; t<= PlasticDuration2; t+= StepSize){
			Agent.EulerStep(StepSize,true);
		}

		// Calculate the Pyloric Fitness
		fit = PyloricPerformance(Agent);

		// Award bonus
		if (fit >= .3){fit = fit+pyloric_boost;}

		// Add to cumulative fitness
		fitness += fit;
	}

    return fitness/(num_points*2);
}

double HPFitnessFunction(TVector<double> &genotype, TMatrix<double> &ptlist, RandomState &rs){
	// Evaluate the fitness of a genotype, wrapping all the mapping and evaluation into one function for TSearch
	
	// Create the agent
	CTRNN Agent(3); //it inherits the plastic pars from the file

	// Instantiate the nervous system
    ifstream ifs;
    ifs.open(circuitfname);
    if (!ifs) {
        cerr << "File not found: " << circuitfname << endl;
        exit(EXIT_FAILURE);
    }
    ifs >> Agent; 
	ifs.close();

	// Map genotype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, genotype.UpperBound());
	GenPhenMapping(genotype, phenotype, Agent.num_pars_changed, Agent.num_neurons_plastic);

	// Instantiate the HP mechanism 
	Agent.SetHPPhenotype(phenotype,StepSize); 

	double fitness = HPPerformance(Agent, ptlist, pyloric_boost);

    return fitness;
}



// ------------------------------------
//  Evolutionary Run Display functions
// ------------------------------------
ofstream Evolfile;
ofstream BestIndividualsFile;

void ResultsDisplay(TSearch &s)
// Store the genotype and phenotype of the best evolved individual, in a format that can be read back in later
{
	TVector<double> bestgenotype(1,s.VectorSize());
	TVector<double> bestphenotype(1,s.VectorSize());
	TVector<int> plasticpars(1,N*N + N); 

	// Extract the parameters of the best individual
	bestgenotype = s.BestIndividual();

	ifstream plasticparsfile;
	plasticparsfile.open(plasticparsfname);
	if (!plasticparsfile) {
		cerr << "File not found: " << plasticparsfname << endl;
		exit(EXIT_FAILURE);
	}
	plasticparsfile >> plasticpars;
	plasticparsfile.close();

	TVector<int> plasticneurons(1,N);
	getNumNeuronsPlastic(plasticneurons, plasticpars, N);
	int num_neurons_plastic = plasticneurons.Sum();

	GenPhenMapping(bestgenotype, bestphenotype, plasticpars.Sum(), num_neurons_plastic);

	BestIndividualsFile << plasticpars << endl <<bestgenotype << endl << bestphenotype << endl << s.BestPerformance() << endl;
}

void EvolutionaryRunDisplay(TSearch &s)
{
	Evolfile << s.Generation() << " " << s.BestPerformance() << " " << s.AvgPerformance() << " " << s.PerfVariance() << endl << s.BestIndividual() << endl;
	//too inefficient to map and store the phenotype in a general way, so just store the genotype
}

// ----------------------------------------------
// The main program - run the evolutionary search
// ----------------------------------------------
int main (int argc, const char* argv[]) 
{	// Evolutionary Output files
	// 1: Output to the current directory
	//--------------------------------------- 
	Evolfile.open("./evol.dat");
	BestIndividualsFile.open("./bestind.dat");
	//--------------------------------------- 

	// 2: Output to a specific (or new) directory
	//--------------------------------------- 
	// char dirname[] = "./Timing Requirements/92/0";
	// int result = mkdir(dirname,0755);
	// Evolfile.open("./Timing Requirements/92/0/evol.dat");
	// BestIndividualsFile.open("./Timing Requirements/92/0/bestind.dat");
	//--------------------------------------- 

	long randomseed = static_cast<long>(time(NULL));
	if (argc > 1){
		randomseed += atoi(argv[1]);
	}

	TVector<int> plasticpars(1,N*N + N);

	ifstream plasticparsfile;
	plasticparsfile.open(plasticparsfname);
	if (!plasticparsfile) {
		cerr << "File not found: " << plasticparsfname << endl;
		exit(EXIT_FAILURE);
	}
	plasticparsfile >> plasticpars;
	plasticparsfile.close();

	TVector<int> plasticneurons(1,N);
	getNumNeuronsPlastic(plasticneurons, plasticpars, N);
	int num_neurons_plastic = plasticneurons.Sum();

	int VectSize = plasticpars.Sum() + (3*num_neurons_plastic); 	
	//Configure search
	TSearch s(VectSize);
	s.SetRandomSeed(randomseed);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	s.SetSelectionMode(RANK_BASED);
	s.SetReproductionMode(GENETIC_ALGORITHM);
	s.SetPopulationSize(POPSIZE);
	s.SetMaxGenerations(GENS);
	s.SetCrossoverProbability(CROSSPROB);
	s.SetCrossoverMode(UNIFORM);
	s.SetMutationVariance(MUTVAR);
	s.SetMaxExpectedOffspring(EXPECTED);
	s.SetElitistFraction(ELITISM);
	s.SetSearchConstraint(1);
	s.SetReEvaluationFlag(0); //  Parameter Variability Modality Only

	// Set Initial Conditions
	// 1: Create a grid of points for initial parameter values
	//-----------------------------------------------------------
	int resolution = 5; //number of values per parameter
	TVector<double> par_vals(1,resolution);
	//manually set the values
	par_vals[1] = -10;
	par_vals[2] = -5;
	par_vals[3] = 0;
	par_vals[4] = 5;
	par_vals[5] = 10;

	int num_pts = pow(resolution,plasticpars.Sum());
	TMatrix<double> ptlist(1,num_pts,1,plasticpars.Sum());
	PointGrid(ptlist,par_vals);
	//-----------------------------------------------------------

	// 2: OR Create a random set of points for initial parameter values
	//-----------------------------------------------------------
	// int num_pts = 50;
	
	// TMatrix<double> ptlist(1,num_pts,1,num);
	// for (int row = 1; row <= num_pts; row++){
	// 	for (int col = 1; col <= plasticpars.Sum(); col++){
	// 		ptlist(row,col) = UniformRandom(-16,16);
	// 	}
	// }
	//-------------------------------------------------------

	s.SetInitialPtsforEval(ptlist);
	s.SetEvaluationFunction(HPFitnessFunction);

	// Execute the search
	s.ExecuteSearch(seed_center_crossing);
	
	Evolfile.close();
	BestIndividualsFile.close();

  return 0;
}
