//----------------------------------------------------------------------------
// For a given B1,B3 plane (centered on pyloric solution), systematically
// explore the parameter space of HP mechanisms (varying LBs) that succeed and fail.
// Other HP metaparameters (range, time constant, window size) are fixed.
// Optionally, run for a while afterwards to check whether ADHP induces a limit cycle (HPLC).
//
//----------------------------------------------------------------------------

#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"
#include "pyloric.h"

// Task params
const double TransientDuration = 50; //Seconds of equilibration with HP off (both before HP starts and before pyloricness is tested)
const double PlasticDuration = 20000; //50000; //Seconds with HP running
const int N = 3;
const int CTRNNphenotypelen = (2*N)+(N*N);

// Things that are the same between ADHP mechanisms
const double range = 0; //assume constant range across neurons
const double Btauval = 150; //right in the middle of evol range
const double SWval = 0; //(in seconds)

const bool HPLC_detection = false;          //whether to run extra time to check for HPLC
const double HPLC_detection_dur = 50;       //how long to continue simulating (in seconds)
const double HPLC_detection_threshold = .5; //how far away in parameter space counts as an ADHP-induced limit cycle

// file that lists which parameters are plastic (1) or not (0):
// --------------------------------------------
//Outer directory mode
const char plasticityparsfname[] = "./plasticpars.dat";
// --------------------------------------------
//Inner directory mode
// const char plasticityparsfname[] = "../../plasticpars.dat";
// --------------------------------------------

// file that lists desired HP metaparameter space resolution. each line corresponds to a dimension
// in the order that they are listed in plasticpars.dat and contains the min, max, and step
// If there are more lines than there are plastic parameters (as specified by plasticpars.dat),
// only reads up to num_plasticpars. If there are too few lines to specify each plastic parameter, 
// generates an error
// --------------------------------------------
//Outer directory mode
char resfname[] = "./metaparres.dat";
// --------------------------------------------
//Inner directory mode
// char resfname[] = "../../metaparres.dat";
// --------------------------------------------

int main(int argc, const char* argv[])
{
    //import and configure plastic pars
	ifstream plasticityparsfile;
	plasticityparsfile.open(plasticityparsfname);
	if (!plasticityparsfile){
		cerr << "File not found: " << plasticityparsfname << endl;
		exit(EXIT_FAILURE);
	}

	TVector<int> plasticitypars(1,N+(N*N));
    TVector<int> plasticneurons(1,N);

	plasticityparsfile >> plasticitypars;
    plasticityparsfile.close();

    getNumNeuronsPlastic(plasticneurons, plasticitypars, N);

    int num_plasticpars = plasticitypars.Sum();
    int num_plasticneurons = plasticneurons.Sum();
    int HPphenotypelen = num_plasticpars + 3*num_plasticneurons;

    //import and configure metaparameter space resolution
    ifstream resfile;
	resfile.open(resfname);
	if (!resfile){
		cerr << "File not found: " << resfname << endl;
		exit(EXIT_FAILURE);
	}

    TMatrix<double> resmat(1,num_plasticpars,1,3);
    resmat.FillContents(0.0);
	TVector<double> parvec(1,num_plasticpars);

	for(int i=1;i<=num_plasticpars;i++){
		for(int j=1;j<=3;j++){ 
			resfile >> resmat(i,j);
		}
        if (resmat(i,3)==0){ //check that every parameter asked for has been specified with nonzero step size
            cerr << "Either not enough resolution parameters given in metaparres.dat OR one is specified with stepsize=0" << endl;
            exit(EXIT_FAILURE);
        }
		parvec(i) = resmat(i,1); //initialize the parameter values at the lowest given bound
	}

    // suggested 2D GRID OF POINTS (spacing in each dimension)
    // --------------------------------------------
    int resolution = 5;
    TVector<double> par_vals(1,resolution);
    par_vals[1] = -10;
    par_vals[2] = -5;
    par_vals[3] = 0;
    par_vals[4] = 5;
    par_vals[5] = 10;
    // --------------------------------------------

    // suggested 3D GRID OF POINTS
    // --------------------------------------------
    // int resolution = 3;
    // TVector<double> par_vals(1,resolution);
    // par_vals[1] = -10;
    // par_vals[2] = 0;
    // par_vals[3] = 10;
    // --------------------------------------------

    // If defined on a grid:
    // --------------------------------------------
    int num_pts = pow(resolution,num_plasticpars);
    TMatrix<double> ptlist(1,num_pts,1,num_plasticpars);
    PointGrid(ptlist,par_vals);
    // --------------------------------------------

    //if want from one specific initial point
    // --------------------------------------------
    // int num_pts = 1;
    // TMatrix<double> ptlist(1,num_pts,1,num_plasticpars);
    // ptlist(1,1) = -1.67349; 
    // ptlist(1,2) = 1.9336;
    // --------------------------------------------

    // Nervous system inputs and slice file outputs
    // --------------------------------------------
    // Outer directory mode
    char infile[] = "./Timing Requirements/86/pyloriccircuit.ns";
    char outfile[] = "./Timing Requirements/86/0/HPparslice.dat";
    char HPLCdetectionfname[] = "./Timing Requirements/86/0/HPLC.dat";
    // --------------------------------------------

    // --------------------------------------------
    // Inner directory Mode
    // char infile[] = "../pyloriccircuit.ns";
    // char outfile[] = "./HPparslice.dat";
    // char HPLCdetectionfname[] = "./HPLC.dat";
    // --------------------------------------------

    ifstream nervoussystemfile;
    nervoussystemfile.open(infile);
    //Define pyloric circuit around which to center the slice
    CTRNN Circuit(3);
    nervoussystemfile >> Circuit;
    nervoussystemfile.close();

    //Define HP parslice output file
    ofstream HPparspacefile;
    HPparspacefile.open(outfile);

    ofstream HPLCfile;
    HPLCfile.open(HPLCdetectionfname);

    //Define HPs based on position in parspace slice
    bool finished = false;
    while (!finished){
        TVector<double> HPphenotype(1,HPphenotypelen);
        int k = 1;
        for (int i=1;i<=num_plasticpars;i++){
            HPphenotype[k] = Btauval;
            k++;
        }
        for (int i=1;i<=num_plasticneurons;i++){
            HPphenotype[k] = parvec[i];
            k++;
        }
        for (int i=1;i<=num_plasticneurons;i++){
            HPphenotype[k] = range;
            k++;
        }
        for (int i=1;i<=num_plasticneurons;i++){
            HPphenotype[k] = SWval;
            k++;
        }

        Circuit.SetHPPhenotype(HPphenotype,StepSize);

        //Check grid of initial points to see how many end up pyloric (will just be counting, not keeping fitness)
        int pyloric_count = 0;
        
        //initialize HPLC detector
        bool HPLC_detected = false;

        //Start from each point on the given grid
        for (int ic=1;ic<=num_pts;ic++){
            for (int paridx=1;paridx<=num_plasticpars;paridx++){
                Circuit.SetArbDParam(paridx,ptlist(ic,paridx));
            }

            //Reset Circuit
            Circuit.RandomizeCircuitOutput(0.5, 0.5);
            Circuit.WindowReset();

            // Run the circuit for an initial transient; HP is off and fitness is not evaluated
            for (double t = StepSize; t <= TransientDuration; t += StepSize) {
                Circuit.EulerStep(StepSize,false);
            }
            
            // Apply plasticity for a period of time
            for (double t = StepSize; t <= PlasticDuration; t += StepSize) {
                Circuit.EulerStep(StepSize,true);
            }

            // Evaluate whether pyloric
            double pyloricness;
            pyloricness = PyloricPerformance(Circuit);

            if (pyloricness >= .3){
                pyloric_count ++;
            }

            //check for HPLC using the parameters set out in preamble
            if(HPLC_detection && !HPLC_detected){
                TVector<double> start_pars(1,num_plasticpars);
                for (int i = 1; i <= num_plasticpars; i++){
                    start_pars[i] = Circuit.ArbDParam(i);
                }
                double distance = 0;
                for (double t=StepSize; t<=HPLC_detection_dur; t+=StepSize){
                    Circuit.EulerStep(StepSize,true);
                    for (int i=1;i<=num_plasticpars;i++){
                        distance += pow(start_pars[i]-Circuit.ArbDParam(i),2);
                    }
                    distance = pow(distance,.5);
                    if (distance > HPLC_detection_threshold){
                        HPLC_detected = true;
                        break;
                    }
                    distance = 0;
                }
            }
        }

        HPparspacefile << pyloric_count << " ";
        if (HPLC_detection){
            HPLCfile << HPLC_detected << " ";
        }

        //and then increase the value of the appropriate parameters
        parvec(num_plasticpars)+=resmat(num_plasticpars,3); //step the last dimension
        for (int i=(num_plasticpars-1); i>=1; i-=1){ //start at the second to last dimension and count backwards to see if the next dimension has completed a run
            if(parvec(i+1)>resmat(i+1,2)){   //if the next dimension is over its max
                HPparspacefile << endl;
                if (HPLC_detection){
                    HPLCfile << endl;
                }
                parvec(i+1) = resmat(i+1,1); //set it to its min
                parvec(i) += resmat(i,3);    //and step the current dimension
            }
        }
        if (parvec(1)>resmat(1,2)){
            finished = true;
        }
    }
    HPparspacefile.close();
    HPLCfile.close();

    return 0;
}