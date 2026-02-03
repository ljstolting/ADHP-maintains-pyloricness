//----------------------------------------------------------------------------
// For a given B1,B3 plane (centered on pyloric solution), systematically
// explore the parameter space of HP mechanisms (LBs,ranges,SWs) that succeed.
// I assume (based on my evolutions) that there will be some planes for which
// it is easy and some for which it is hard or impossible. And intermediates for
// which there are two types of mechanisms that work. 
//----------------------------------------------------------------------------

#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"
#include "pyloric.h"

//INDIVIDUALIZED HP MODE
#include <stdio.h>
#include <cstring>
#include <sys/stat.h>
#define Max_Digits 2

// Task params
const double TransientDuration = 25; //Seconds of equilibration with HP off (both before HP starts and before pyloricness is tested)
const double PlasticDuration = 20000; //50000; //Seconds with HP running
const int N = 3;
const int CTRNNphenotypelen = (2*N)+(N*N);
// const int num = 2; //how many paramerters will be under HP control, and perturbed
const int indiv_idx = 0; //index of the first individual you'll run, only used if file is list
const int num_indivs = 1; //how many genomes in the file (100 for Local Run Mode)
const bool shiftedrho = true;

const bool HPLC_detection = false;
const double HPLC_detection_dur = 50;       //how long to continue simulating to check for HPLC (in seconds)
const double HPLC_detection_threshold = .5;   //how far away in parameter space counts as an ADHP-induced limit cycle

//file that lists the dimensions HP operates in
//LOCAL MODE
char dimsfname[] = "./metapardimensions2D.dat";
//SUPERCOMPUTER MODE
// char dimsfname[] = "../../metapardimensions2D.dat";

// file that lists HP parameter space specifications -- Bounds differ between HP mechanisms
//LOCAL MODE
char resfname[] = "./metaparres2D_Q4.dat";
//SUPERCOMPUTER MODE
// char resfname[] = "../../metaparres2D.dat";

// Things that are the same between ADHP mechanisms
const double range = 0; //assume constant range across neurons
const double Btauval = 150; //right in the middle of evol range
const double SWval = 0; //(in seconds)

const bool range_encoding = true;

int main(int argc, const char* argv[])
{
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

	TVector<int> dims(1,N+(N*N));
	int num;

	dimsfile >> dims;
	num = dims.Sum();

    int HPphenotypelen = num*4;

    TMatrix<double> resmat(1,num,1,3);
	TVector<double> parvec(1,num);

	for(int i=1;i<=num;i++){
		for(int j=1;j<=3;j++){ //each line contains the min, max, and step
			resfile >> resmat(i,j);
		}
		parvec(i) = resmat(i,1); //initialize the parameter values at the lowest given bound
	}

    // 2D GRID OF POINTS (spacing in each dimension)
    int resolution = 5;
    TVector<double> par_vals(1,resolution);
    par_vals[1] = -10;
    par_vals[2] = -5;
    par_vals[3] = 0;
    par_vals[4] = 5;
    par_vals[5] = 10;

    // 3D GRID OF POINTS
    // int resolution = 3;
    // TVector<double> par_vals(1,resolution);
    // par_vals[1] = -10;
    // par_vals[2] = 0;
    // par_vals[3] = 10;

    int num_pts = pow(resolution,num);
    TMatrix<double> ptlist(1,num_pts,1,num);
    PointGrid(ptlist,par_vals);

    //if want from one initial point
    // int num_pts = 1;
    // TMatrix<double> ptlist(1,num_pts,1,num);
    // ptlist(1,1) = -1.67349; 
    // ptlist(1,2) = 1.9336;

    ifstream ifs;

    // Local Run Mode
    // char fname[] = "../Pyloric CTRNN Genomes/goodgenomesnormalizedfit.dat";
    // ifs.open(fname);
    // if (!ifs) {
    //     cerr << "File not found: " << fname << endl;
    //     exit(EXIT_FAILURE);
    // }

    // TVector<double>phenotype(1,CTRNNphenotypelen);
    // double pyl_fitness =  0;

    // //Resuming search after a segfault
    // for(int bar=0;bar<10;bar++){
    //     ifs >> phenotype;
    //     ifs >> pyl_fitness;
    // }

    for (int indiv=indiv_idx;indiv<indiv_idx+num_indivs;indiv++){

        //Define pyloric circuit around which to center the slice
        CTRNN Circuit(3);

        // One circuit only mode
        ifs.open("./Specifically Evolved HP mechanisms/Every Circuit/86/pyloriccircuit.ns");

        // Parallel Supercomputer Mode
        // ifs.open("./pyloriccircuit.ns");

        ifs >> Circuit;

        // Local Serial Run Mode (one file)
        // ifs >> phenotype;
        // ifs >> pyl_fitness;
        // phenotype >> Circuit;

        Circuit.SetPlasticityPars(dims);
        Circuit.ShiftedRho(shiftedrho);

        //Define HP parslice output file
        ofstream HPparspacefile;

        // OUTPUT FILES
        // Local Serial Run Mode 
        // char indiv_char[Max_Digits + sizeof(char)];
        // std::sprintf(indiv_char, "%d", indiv);
            
        // char outfile[(Max_Digits) + (sizeof(char)) + 51+18];
        // strcpy(outfile, "./Specifically Evolved HP mechanisms/Every Circuit/");
        // strcat(outfile, indiv_char);
        // strcat(outfile, "/HPparslice.dat");

        // Parallel Supercomputer Mode OR 
        // char outfile[] = "./HPparslice_newrho_res5_HPofftest.dat";
        // char outfile[] = "./HPparslice_3D_res3_highres.dat";

        // Only one cicuit/HP pair mode
        char outfile[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/HPparspace_Q4.dat";

        HPparspacefile.open(outfile);
        ifs.close();

        ofstream HPLCfile;

        //Supercomputer Mode
        // char HPLCdetectionfname[] = "./HPLC.dat";
        //Local mode
        char HPLCdetectionfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/86/HPparspace_Q4_HPLC.dat";

        HPLCfile.open(HPLCdetectionfname);

        //Define HPs based on position in parspace slice
        bool finished = false;
	    while (!finished){
            TVector<double> HPphenotype(1,HPphenotypelen);
            int k = 1;
            for (int i=1;i<=num;i++){
                HPphenotype[k] = Btauval;
                k++;
            }
            for (int i=1;i<=num;i++){
                HPphenotype[k] = parvec[i];
                k++;
            }
            for (int i=1;i<=num;i++){
                HPphenotype[k] = range;
                k++;
            }
            for (int i=1;i<=num;i++){
                HPphenotype[k] = SWval;
                k++;
            }

            Circuit.SetHPPhenotype(HPphenotype,StepSize,true);

            cout << Circuit.PlasticityLB(1) << " " << Circuit.PlasticityLB(2) << " " << Circuit.PlasticityLB(3) << " " << Circuit.PlasticityUB(1) << " " << Circuit.PlasticityUB(2) << " " << Circuit.PlasticityUB(3) << endl;
            // cout << "after phen set" << endl;
            //Check grid of initial points to see how many end up pyloric (will just be counting, not keeping fitness)
            int pyloric_count = 0;
            
            //initialize HPLC detector
            bool HPLC_detected = false;

            //Start from each point on the given grid
            for (int ic=1;ic<=num_pts;ic++){
                for (int paridx=1;paridx<=num;paridx++){
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
                // cout << Circuit.biases << endl;
                // Evaluate whether pyloric
                double pyloricness;
                // cout << "before pyl perf" << endl;
                pyloricness = PyloricPerformance(Circuit); // test pyloricness with HP still on
                // cout << "after pyl perf" << endl;
                if (pyloricness >= .3){
                    pyloric_count ++;
                }

                //check for HPLC using the parameters set out in preamble
                if(HPLC_detection && !HPLC_detected){
                    TVector<double> start_pars(1,num);
                    for (int i = 1; i <= num; i++){
                        start_pars[i] = Circuit.ArbDParam(i);
                    }
                    double distance = 0;
                    for (double t=StepSize; t<=HPLC_detection_dur; t+=StepSize){
                        Circuit.EulerStep(StepSize,true);
                        for (int i=1;i<=num;i++){
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
            cout << pyloric_count << endl; 
            if (HPLC_detection){
                HPLCfile << HPLC_detected << " ";
            }

            //and then increase the value of the appropriate parameters
            parvec(num)+=resmat(num,3); //step the last dimension
            for (int i=(num-1); i>=1; i-=1){ //start at the second to last dimension and count backwards to see if the next dimension has completed a run
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
    }

    return 0;
}