// --------------------------------------------------------------
//  Evolve a Pyloric-like CTRNN around which to center the slices
//  Current version does not seem quite right... must have been successful
//  at some point to create savinggoodgenomes file
// --------------------------------------------------------------
#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"
#include "pyloric.h"

//#define PRINTOFILE

// Task params
const double TransientDuration = 500; //in seconds
// const double RunDuration = 250; //in seconds
// const double StepSize = 0.01;
// const int RunSteps = RunDuration/StepSize; // in steps

// Detection params
// const double burstthreshold = .5; //threshold that must be crossed for detecting bursts
// const double tolerance = .05; //for detecting double periodicity

// EA params
const int POPSIZE = 100;
const int GENS = 500;
const int trials = 200;        		//max number of times to run the EA from random starting pop
const int num_successes = 1; 		//evolve until you have x successful circuits
const double success_fitness = 5; //a successful evolution is defined as >= this value
const double MUTVAR = 0.1;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;

// Parameter variability modality only
//const int Repetitions = 10; 
//const int AnalysisReps = 100;

// Nervous system params
const int N = 3;
const double WR = 16.0; 
const double BR = 16.0; //(WR*N)/2; //<-for allowing center crossing
const double TMIN = .1; 
const double TMAX = 2; 

// Plasticity parameters
const int WS = 120;		// Window Size of Plastic Rule (in steps size) (so 0 is no window)
const double B = 0.1; 		// Plasticity Low Boundary (symmetric)
const double BT = 20.0;		// Bias Time Constant
const double WT = 40.0;		// Weight Time Constant

int	CTRNNVectSize = N*N + 2*N;

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				phen(k) = MapSearchParameter(gen(k), -WR, WR);
				k++;
			}
	}
}


double PyloricFitnessFunction(TVector<double> &genotype, TMatrix<double> &ptlist, RandomState &rs){
	TVector<double> phenotype;
	phenotype.SetBounds(1, CTRNNVectSize);
	GenPhenMapping(genotype, phenotype);

	CTRNN Agent(3);

	// Instantiate the nervous system
	phenotype >> Agent;
	// cout << Agent.taus << " " << Agent.biases << " " << Agent.weights << endl;
	Agent.RandomizeCircuitState(0,0);

	for (double t=StepSize;t<=TransientDuration;t+=StepSize){
		// cout << Agent.outputs << endl;
		Agent.EulerStep(StepSize,false);
	}
	// cout << Agent.outputs << endl;
	
	double fitness = PyloricPerformance(Agent);
	return fitness;
}

// ------------------------------------
// Display functions
// ------------------------------------
ofstream Evolfile;
ofstream BestIndividualsFile;
ofstream BestoftheBestFile;

void EvolutionaryRunDisplay(TSearch &s)
{
	
	//cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
	Evolfile << s.Generation() << " " << s.BestPerformance() << " " << s.AvgPerformance() << " " << s.PerfVariance() << endl;

	TVector<double> bestVector;
	TVector<double> phenotype;
	phenotype.SetBounds(1, CTRNNVectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	GenPhenMapping(bestVector, phenotype);

	Evolfile << phenotype << endl;
}
int trial = 1;
void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	TVector<double> phenotype;
	phenotype.SetBounds(1, CTRNNVectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	GenPhenMapping(bestVector, phenotype);

	BestIndividualsFile << trial << endl;
	BestIndividualsFile << bestVector << endl << phenotype << endl;
	BestIndividualsFile << s.BestPerformance() << endl << endl;

	cout << trial << "finished" << endl;

	trial ++;
}
void BestRecord(TSearch &s,ofstream &bestofthebestfile)
{
	TVector<double> bestVector;
	TVector<double> phenotype;
	phenotype.SetBounds(1, CTRNNVectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	GenPhenMapping(bestVector, phenotype);

	bestofthebestfile << bestVector << endl << phenotype << endl;
	bestofthebestfile << s.BestPerformance() << endl << endl;
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) 
{
// Evolution condition
	Evolfile.open("./pyloric_evol_timing_legacy.dat");
	BestIndividualsFile.open("./pyloric_bestind_timing_legacy.dat");
	BestoftheBestFile.open("./pyloric_goodgenomes_timing_legacy.dat");

	int success = 0;
	for (int i=1;i<=trials;i++){
		if(success < num_successes){
			long randomseed = static_cast<long>(time(NULL));
			if (argc == 2)
				randomseed += atoi(argv[1]);
			// long IDUM=-time(0);
			TSearch s(CTRNNVectSize);

			// Configure the search
			s.SetRandomSeed(randomseed);
			//cout << IDUM << endl;
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
			s.SetEvaluationFunction(PyloricFitnessFunction);
			s.ExecuteSearch(true); //Center Crossing Mode or not
			double bestperf = s.BestPerformance();
			if (bestperf>=success_fitness){
				success++;
				BestRecord(s,BestoftheBestFile);
			}
		}
		else{break;}
	}
	Evolfile.close();
	BestIndividualsFile.close();
	BestoftheBestFile.close();
}
