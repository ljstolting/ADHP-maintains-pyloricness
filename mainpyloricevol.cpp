// --------------------------------------------------------------
//  Evolve a Pyloric-like CTRNN around which to center the slices
//  
// --------------------------------------------------------------

#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"
#include "pyloric.h"
#include "VectorMatrix.h"

//#define PRINTOFILE

// Task params
const double TransientDuration = 50; //in seconds

// EA params
const int POPSIZE = 50;
const int GENS = 500;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.0;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;
const bool seed_center_crossing = false; //initialize the population with center-crossing networks?

// Nervous system params and allowed ranges
const int N = 3;
const double WR = 16.0; 
const double BR = 16.0; //(WR*N)/2; //<-for allowing center crossing
const double TMIN = .1; 
const double TMAX = 2; 

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
	
	// initialize circuit outputs at .5
	Agent.RandomizeCircuitOutput(.5,.5);

	// Run the circuit for an initial transient; fitness is not yet evaluated
	for (double t=StepSize;t<=TransientDuration;t+=StepSize){
		Agent.EulerStep(StepSize,false);
	}
	
	double fitness = PyloricPerformance(Agent);
	return fitness;
}

// ------------------------------------
// Display functions
// ------------------------------------
ofstream Evolfile;
ofstream BestFile;
ofstream BestTrajectoryFile;
ofstream BestPyloricBurstsFile;

void EvolutionaryRunDisplay(TSearch &s)
{
	Evolfile << s.Generation() << " " << s.BestPerformance() << " " << s.AvgPerformance() << " " << s.PerfVariance() << endl;

	TVector<double> bestVector;
	TVector<double> phenotype;
	phenotype.SetBounds(1, CTRNNVectSize);

	// Track the genotype of the best individual
	bestVector = s.BestIndividual();
	GenPhenMapping(bestVector, phenotype);

	Evolfile << phenotype << endl;
	return;
}
void ResultsDisplay(TSearch &s)
{
	TVector<double> bestVector;
	TVector<double> phenotype;
	phenotype.SetBounds(1, CTRNNVectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	GenPhenMapping(bestVector, phenotype);

	CTRNN BestAgent(3);
	phenotype >> BestAgent;

	BestFile << BestAgent;
	double fit = PyloricPerformance(BestAgent, BestTrajectoryFile, BestPyloricBurstsFile);
	return;
}

// ------------------------------------
// The main program
// ------------------------------------
int main (int argc, const char* argv[]) 
{
	// Evolutionary output files
	Evolfile.open("./evol.dat");
	BestFile.open("./pyloriccircuit.ns");
	BestTrajectoryFile.open("./pylorictrajectory.ns");
	BestPyloricBurstsFile.open("./pyloricbursttimes.dat");

	long randomseed = static_cast<long>(time(NULL));
	if (argc > 1){randomseed += atoi(argv[1]);}

	TSearch s(CTRNNVectSize);

	// Configure the search
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
	s.SetReEvaluationFlag(0); 

	s.SetEvaluationFunction(PyloricFitnessFunction);

	s.ExecuteSearch(seed_center_crossing); //Center Crossing Mode or not

	Evolfile.close();
	BestFile.close();
}
