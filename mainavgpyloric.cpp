//------------------------------------------------------
// Gather data needed to predict ADHP performance
// Simultaneously calculates each neuron's average value (see expression 7)
// and the pyloric fitness of each circuit defined on a specified slice/volume
// in parameter space, centered at an evolved pyloric solution
//
//  ****This script should be run within the directory of the evolved pyloric circuit around which 
//  you are centering the slice/volume****
//
//  Ensure the ./plasticpars.dat file reflects the dimensions around which
//  you want to extend your search. Ensure the ./res.dat file relfects 
//  the resolution with which you want to sample CTRNN parameter space
//  (dimensions ordered in the same way as the ./plasticpars.dat file)
// 
//  You may run this script in one of three modes: 
//      1. Full data return mode - Outputs a file containing average values for relevant neurons
//          and another containing pyloric fitness, for every circuit configuration. 
//          Predicted ADHP performance must be calculated separately by coarse graining averages.
//          Useful in 2D to obtain pyloric fitness and average slices.
//
//      2. Online processing mode - Performs coarse-graining online and tracks the number of pyloric
//          and non-pyloric circuit configurations which display each combination of averages (rounded). 
//          Detection of pyloricness is not always perfect and the number of configurations in each
//          averaging category varies widely, so it is best to consider ratio of pyloric:nonpyloric.
//          Particularly useful in 3 or more dimensions.
//              *Ensure the ./metaparres.dat file reflects the resolution with which
//              you want to coarse-grain the space of average neural values
//
//      3. Audit mode - For particular points in ADHP metaparameter space where predictions don't align
//          with simulations, output details (parameters, pyloricness) about the circuit configurations 
//          which exhibit the average values of interest (rounded) WITHOUT having to record this data 
//          about all the circuit configurations in the space (again, particularly useful in 3D)
//              *Ensure the ./pt_of_interest.dat file specifies the point you want to investigate.
//  
//  Additionally, to manage data volume and compute time in any of these modes (particularly in 3D),
//  you can specify a subset of the total parameter space to check, allowing you to 
//  search these subsets in parallel. To do this, execute the program from within a 
//  sub-directory and FOLLOW YOUR CALL WITH AN ARGUMENT which is the integer index 
//  of the section you want to sample. Subsets will be broken along the first dimension
//  in steps of width 'slice_step'*(resolution of first dimension). The relevant output  
//  files appear in specified sub-directory and data can be combined with others later.
//  
//------------------------------------------------------

#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"
#include "pyloric.h"

//UNCOMMENT DESIRED MODE:
//----------------------------------------------------------------
// 1: FULL DATA RETURN MODE
bool record_all = true;
bool audit_mode = false;

// Output files
char out1fname[] = "./HPAgnosticAverageSlice.dat"; //track average at each point
char out2fname[] = "./PyloricPerformanceSlice.dat"; //track pyloric fitness at each point
//----------------------------------------------------------------

//----------------------------------------------------------------
// // 2: ONLINE PROCESSING MODE
// bool record_all = false;
// bool audit_mode = false;

// // output files
// char out1fname[] = "./numPyloricConfigs.dat"; 
//     //number pyloric circuits rounded to each pt on lattice
// char out2fname[] = "./numNonpyloricConfigs.dat"; 
//     //number non-pyloric circuits rounded to each pt on lattice
//----------------------------------------------------------------

//----------------------------------------------------------------
// // 3: AUDIT MODE
// bool record_all = false;
// bool audit_mode = true;

// // output files
// char out1fname[] = "./relevant_pt_features.dat";      //biases, exact avgs for each circuit
// char out2fname[] = "./relevant_pt_pyloricness.dat";   //pyloric fitness, pyloric features for each circuit
//----------------------------------------------------------------

// COBBLED PARALLELISM - not used if no argument provided after execution call
const int slice_step = 2; //how many steps (in the first dimension) to span for each run

// Input files
char resfname[] = "../../res.dat";  
char dimsfname[] = "../../plasticpars.dat";
char metaparresfname[] = "../../metaparres.dat";        //only used in online processing mode
char ptofinterestfname[] = "../../pt_of_interest.dat";  //only used in audit mode
char circuitfname[] = "./pyloriccircuit.ns";

// Task params
const double TransientDuration = 50; //seconds without HP 
const double LeaveDuration = 1; //max seconds to allow the circuit states to leave the bubble before we assume it's at a fixed point
const double leaving_tolerance = 5*StepSize; //how far away in state space counts as leaving
const double RunDuration = 10; //max seconds to look for a single oscillation cycle
const double return_tolerance = 4*StepSize; //must be less than leaving tolerance
int max_steps = int((RunDuration*4)/StepSize); //how much memory to allocate for the max size outputhistory

const int N = 3;

int main (int argc, const char* argv[]) 
{
    // Open input files
    ifstream resfile;
	resfile.open(resfname);
	if (!resfile){
		cerr << "File not found: " << resfname << endl;
		exit(EXIT_FAILURE);
	}
	ifstream dimsfile;
	dimsfile.open(dimsfname);
	if (!dimsfile){
		cerr << "File not found: " << dimsfname << endl;
		exit(EXIT_FAILURE);
	}
    ifstream metaparresfile;
    metaparresfile.open(metaparresfname);
    if (!metaparresfile){
        cerr << "File not found:" << metaparresfname << endl;
        exit(EXIT_FAILURE);
    }
    ifstream ptofinterestfile;
    ptofinterestfile.open(ptofinterestfname);
    if (!ptofinterestfile) {
        cerr << "File not found: " << ptofinterestfname << endl;
        exit(EXIT_FAILURE);
    }
    ifstream circuitfile;
    circuitfile.open(circuitfname);
    if (!circuitfile) {
        cerr << "File not found: " << circuitfname << endl;
        exit(EXIT_FAILURE);
    }

	// Create files to hold data
	ofstream outfile1;
	outfile1.open(out1fname);
    ofstream outfile2;
	outfile2.open(out2fname);

    // Organize information about the grid of points, gathered from the res file
    TVector<int> dims(1,N+(N*N));
	int num_dims;

	dimsfile >> dims;
    dimsfile.close();
	num_dims = dims.Sum();

	TMatrix<double> resmat(1,num_dims,1,3);
	TVector<double> parvec(1,num_dims);

	for(int i=1;i<=num_dims;i++){
		for(int j=1;j<=3;j++){
			resfile >> resmat(i,j);
		}
	}
    resfile.close();

    // if there is an argument present in executable call, 
    // no longer actually takes first dim from resfile but adjusts based on argument given
    // uses first dimension step specified by res file, and ensures that no calculations are repeated
    if (argc>1){
        int slicenum = atoi(argv[1]);
        resmat(1,1) = resmat(1,1)+(slicenum*slice_step*resmat(1,3)); //slicenum should have zero-indexing
        resmat(1,2) = resmat(1,1)+((slice_step-1)*resmat(1,3)); //resmat upper bound is inclusive
    }   
    cout << "First dimension stepping between:" << resmat(1,1) << " and " << resmat(1,2) << endl;

    for(int i=1;i<num_dims;i++){
		parvec(i) = resmat(i,1); //initialize the parameter values at the lowest given bound
    }

    // Create CTRNN and load in parameters
    CTRNN Circuit(N);
    circuitfile >> Circuit;
    circuitfile.close();

    Circuit.SetPlasticityPars(dims); //so we can use HP functionality to keep track of the parameters that are being varied
    int regulated_neurons = Circuit.plasticneurons.Sum();

    TMatrix<double> metaparresmat(1,regulated_neurons,1,3);
    for (int i=1;i<=regulated_neurons;i++){
        for (int j=1;j<=3;j++){
            metaparresfile >> metaparresmat(i,j);
        }
    }
    metaparresfile.close();

    //if online processing mode, define a matrix (which will actually be represented as a vector)
    // on which to keep track of pyloric and nonpyloric circuit counts
    TVector<int> metaparres_lengths(1,regulated_neurons);
    int total_idx_len = 1;
    for (int i=1; i<=regulated_neurons; i++){
        int idxlen = (metaparresmat(i,2) - metaparresmat(i,1))/metaparresmat(i,3);
        idxlen = idxlen+1; //upper bound is inclusive
        metaparres_lengths(i) = idxlen;
        total_idx_len = total_idx_len * idxlen;
    }

    TVector<int> pyloriccount(1,total_idx_len);
    pyloriccount.FillContents(0);
    TVector<int> nonpyloriccount(1,total_idx_len);
    nonpyloriccount.FillContents(0);

    // if audit mode, load in (and round if necessary) the metaparres pt of interest and then convert to metaparres matrix indecies
    TVector<int> ptofinterest_idx(1,regulated_neurons);
    for (int i=1; i<=regulated_neurons; i++){
        double v;
        ptofinterestfile >> v;
        v = v-metaparresmat(i,1); //redundant because i can't see any reason why metaparres would start past 0
        int idx = round(v/metaparresmat(i,3));
        ptofinterest_idx(i) = idx;
    }
    ptofinterestfile.close();

    // Set some variables that will be used in the loop
    TVector<double> acc(1,N); //vector to store the value of the proxy expression for each neuron
    bool finished = false;

    // And finally, start the calculations
	while (!finished){
        // set the parameters to the new grid value and reset the neural states
		for (int i=1;i<=num_dims;i++){
			Circuit.SetArbDParam(i,parvec(i));
		}

        for (int neuron=1; neuron <= N; neuron ++){
			Circuit.SetNeuronOutput(neuron,.5); //puts in line with the way the average/proxy is calculated most cleanly
		}
		Circuit.WindowReset();

        //pass transient
        for (double t=StepSize;t<=TransientDuration;t+=StepSize){
            Circuit.EulerStep(StepSize,0);
        }
        // new time-saving and consolidating structure: one cycle, calculate average, if average is of interest, then roughly three more cycles, calc pyloric fitness
        TVector<double> startstate(1,N);
        TVector<double> avg(1,N);
        avg.FillContents(0.0);
        TMatrix<double> outputhist(1,max_steps,1,N);
        int stepnum = 0;
        double t = StepSize; 
        double dist = 0;
        bool left,returned;
        left = false;
        returned = false;
        for (int i = 1; i <= N; i++){
            startstate[i] = Circuit.NeuronState(i);
        }
        while ((t<=LeaveDuration) && (left == false)){
            Circuit.EulerStep(StepSize,0);
            stepnum ++;
            t += StepSize;
            dist = 0;
            for (int i = 1; i <= N; i++){
                dist += pow(Circuit.NeuronState(i)-startstate[i],2);
                avg[i] += Circuit.NeuronOutput(i); 
                //record in output hist
                outputhist(stepnum,i) = Circuit.NeuronOutput(i);
            }
            dist = sqrt(dist);

            if ((dist > leaving_tolerance)){
                left = true;
            }
        }
        if (left == true){ //if hasn't left by now, assume to be at equilibrium
            while ((t<=RunDuration) && (returned == false)){
                Circuit.EulerStep(StepSize,0);
                stepnum ++;
                t += StepSize;
                dist = 0;
                for (int i = 1; i <= N; i++){
                    dist += pow(Circuit.NeuronState(i)-startstate[i],2);
                    avg[i] += Circuit.NeuronOutput(i); 
                    //record in output hist
                    outputhist(stepnum,i) = Circuit.NeuronOutput(i);
                }
                dist = sqrt(dist);

                if (dist < return_tolerance){
                    returned = true;
                }
            }
        }
        //calculate average
        for (int i = 1;i <= N;i++){
            avg[i] = avg[i]/stepnum;
        }

        //calculate the metapar res index if needed
        TVector<int> add_idx(1,regulated_neurons);
        add_idx.FillContents(1);
        if(audit_mode || !record_all){
            //derive the index of the matrix that corresponds with the point's average values
            for (int i=1; i<=regulated_neurons; i++){
                double v = avg(i)-metaparresmat(i,1);
                int idx = round(v/metaparresmat(i,3));
                add_idx(i) = idx;
            }
        }

        bool cont = true;
        if(audit_mode){
            for (int i=1; i<=regulated_neurons; i++){
                if (add_idx(i) != ptofinterest_idx(i)){
                    cont = false;
                }
            }
        }
        if(cont){
            if (audit_mode){outfile1 << Circuit.biases << endl << avg << endl << endl;}
            //if oscillitory, allow time for roughly three more cycles
            if (left){
                int first_cycle_stepnum = stepnum;
                for (int i=1;i<=3*first_cycle_stepnum;i++){
                    //run for three more cycles, to ensure 3 PD starts
                    stepnum ++;
                    Circuit.EulerStep(StepSize,0);
                    for (int i = 1; i <= N; i++){
                        outputhist(stepnum,i) = Circuit.NeuronOutput(i);
                    }
                }
            }
        
            //take only the part of the output hist that actually ended up getting filled
            TMatrix<double> newoutputhist(1,stepnum,1,N);
            for (int i=1;i<=stepnum;i++){
                for (int j=1;j<=N;j++){
                    newoutputhist(i,j) = outputhist(i,j);
                }
            }

            double pyl_fitness = 0;
            TVector<double> featuresvec(1,8);
            featuresvec.FillContents(0);
            if (left){
                BurstTimesfromOutputHist(newoutputhist,featuresvec);
                pyl_fitness = PyloricFitFromFeatures(featuresvec);
            }
            if (audit_mode){
                outfile2 << featuresvec << endl << endl;
            }
            // ALL DATA MODE: record all data into files
            else if (record_all){
                outfile1 << avg << endl;
                outfile2 << pyl_fitness << " ";
            }
            else{
                // ONLINE PROCESSING MODE: record the number of pyloric and nonpyloric circuits with average values closest to each point
                // convert add_idx to flattened version for recording
                int flattened_idx = 1;
                for(int i = 1; i <= regulated_neurons; i++){
                    int multiplier = 1;
                    for(int j = i+1; j <= regulated_neurons; j++){
                        multiplier *= metaparres_lengths(j);
                    }
                    flattened_idx += add_idx(i)*multiplier;
                }
                //step appropriate matrix
                if (pyl_fitness>=.3){
                    pyloriccount(flattened_idx) ++;
                }
                else{
                    nonpyloriccount(flattened_idx) ++;
                }
            }
        }

		parvec(num_dims)+=resmat(num_dims,3); //step the last dimension
		for (int i=(num_dims-1); i>=1; i-=1){ //start at the second to last dimension and count backwards to see if the next dimension has completed a run
			if(parvec(i+1)>resmat(i+1,2)){   //if the next dimension is over its max
				parvec(i+1) = resmat(i+1,1); //set it to its min
				parvec(i) += resmat(i,3);    //and step the current dimension
                if (record_all){
                    outfile1 << endl;
                    outfile2 << endl;
                }
			}
		}
		if (parvec(1)>resmat(1,2)){
			finished = true;
        }
    }
    //Output the pyloric and nonpyloric counts for online processing mode all at once, but restructure like a matrix
    if (!record_all && !audit_mode){
        int k = 1;
        TVector<int> dim_tracker(1,regulated_neurons);
        dim_tracker.FillContents(1);
        while (dim_tracker(1)<=metaparres_lengths(1)){
            outfile1 << pyloriccount[k] << " ";
            outfile2 << nonpyloriccount[k] << " ";
            k++;
            for (int i = regulated_neurons-1; i>=1; i-=1){
                if (dim_tracker(i+1) > metaparres_lengths(i)){
                    dim_tracker(i+1) = 1;
                    dim_tracker(i) ++;
                    outfile1 << endl;
                    outfile2 << endl;
                }
            }
        }
    }
    //close output files
    outfile1.close();
    outfile2.close();

    return 0;
}