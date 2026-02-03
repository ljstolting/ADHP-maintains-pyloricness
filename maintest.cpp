//----------------------------------------------------------------------------
// Test a given evolved HP mechanism on all of the evolved pyloric solutions 
// in the "savinggoodgenomes" file
// combine with mainrunpts?
//----------------------------------------------------------------------------

#include "TSearch.h"
#include "CTRNN.h"
#include "random.h"
#include "pyloric.h"

#include <stdio.h>
#include <cstring>
#include <sys/stat.h>
#define Max_Digits 2

// Task params
const double TransientDuration = 50; //Seconds with HP off
const double PlasticDuration = 5000; //Seconds with HP running between each check
const int num_checks = 5;             //How many times to check the pyloric fitness, per initial condition (to examine whether it's reached a steady state)
const int N = 3;
const int CTRNNphenotypelen = (2*N)+(N*N);
const int num_indivs = 1; //how many genomes to loop through
const int num_evols = 211; //how many HP evolutions per circuit
const int num_dims = 2; //how many dimensions does HP operate in (equivalently, how many dimensions is the plane you are testing)

int main(int argc, const char* argv[])
{
    // 2D GRID OF POINTS
    int resolution = 11;
    TVector<double> par_vals(1,resolution);
    // par_vals[1] = -15;
    // par_vals[2] = -9;
    // par_vals[3] = -3;
    // par_vals[4] = 3;
    // par_vals[5] = 9;
    // par_vals[6] = 15;

    par_vals[1] = -15;
    par_vals[2] = -12;
    par_vals[3] = -9;
    par_vals[4] = -6;
    par_vals[5] = -3;
    par_vals[6] = 0;
    par_vals[7] = 3;
    par_vals[8] = 6;
    par_vals[9] = 9;
    par_vals[10] = 12;
    par_vals[11] = 15;


    int num_pts = pow(resolution,num_dims);
    TMatrix<double> ptlist(1,num_pts,1,num_dims);
    PointGrid(ptlist,par_vals);

    bool range_encoding = true;
    // char outdirectory[52] = "./Specifically Evolved HP mechanisms/Every Circuit/";
    char outdirectory[46] = "./Specifically Evolved HP mechanisms/Pete/2D/";
    // char circuitfname[19] = "/pyloriccircuit.ns";
    char circuitfname[] = "./Specifically Evolved HP mechanisms/Every Circuit/19/pyloriccircuit.ns";
    char evolvedHPfname[13] = "/bestind.dat";
    char testfname[27] = "/recoverytest.dat";

    CTRNN Circuit(N);

    for (int indiv = 0; indiv < num_indivs; indiv ++){

        ifstream circuitfile;

        char indiv_char[Max_Digits + sizeof(char)];
        std::sprintf(indiv_char, "%d", indiv);

        // char circuitfilename[Max_Digits + sizeof(char) + 51 + 18];
        // strcpy(circuitfilename, outdirectory);
        // strcat(circuitfilename, indiv_char);
        // strcat(circuitfilename,circuitfname);
        // circuitfile.open(circuitfilename);
        circuitfile.open(circuitfname);
        if (!circuitfile) {
            // cerr << "File not found: " << circuitfilename << endl;
            cerr << "File not found: " << circuitfname << endl;
            exit(EXIT_FAILURE);
        }
        circuitfile >> Circuit;
        Circuit.ShiftedRho(false);
        circuitfile.close();

        for (int HPevol = 0; HPevol < num_evols; HPevol++){

            ofstream testfile;
            ifstream evolvedHPfile;

            char testfilename[Max_Digits + sizeof(char) + Max_Digits + sizeof(char) + 51 + 1 + 26];
            strcpy(testfilename, outdirectory);
            // strcat(testfilename, indiv_char);

            // cout << testfilename << endl;

            char evolvedHPfilename[Max_Digits + sizeof(char) + sizeof(char) + 51 + 1 + 12];
            strcpy(evolvedHPfilename, outdirectory);
            // strcat(evolvedHPfilename, indiv_char);

            char evol_char[Max_Digits + sizeof(char)];
            std::sprintf(evol_char, "%d", HPevol);

            // strcat(testfilename, "/");
            strcat(testfilename, evol_char);
            strcat(testfilename,testfname);
            testfile.open(testfilename);
            if (!testfile) {
                cerr << "No target directory found for: " << testfilename << endl;
                exit(EXIT_FAILURE);
            }

            // strcat(evolvedHPfilename, "/");
            strcat(evolvedHPfilename, evol_char);
            strcat(evolvedHPfilename,evolvedHPfname);

            // cout << Circuit.CircuitSize() << endl;

            // cout << evolvedHPfilename << endl;
            evolvedHPfile.open(evolvedHPfilename);

            if (!evolvedHPfile) {
                cerr << "File not found: " << evolvedHPfilename << endl;
                exit(EXIT_FAILURE);
            }

            Circuit.SetHPPhenotypebestind(evolvedHPfile,StepSize,range_encoding);
            evolvedHPfile.close();

            //IN THE BIAS1,BIAS3 PLANE
            for (int i=1;i<=num_pts;i++){
                Circuit.SetNeuronBias(1,ptlist(i,1));
                Circuit.SetNeuronBias(3,ptlist(i,2));
                Circuit.RandomizeCircuitState(0, 0);

                // Run the circuit for an initial transient; HP is off and fitness is not evaluated
                for (double t = StepSize; t <= TransientDuration; t += StepSize) {
                    Circuit.EulerStep(StepSize,false);
                }

                for (int j=1; j<=num_dims; j++){
                    testfile << ptlist(i,j) << " ";
                }

                // Run, check, and record pyloricness for as many times specified
                for (int check=1; check<=num_checks; check++){
                    for (double t = StepSize; t <= PlasticDuration; t += StepSize) {
                        Circuit.EulerStep(StepSize,true);
                    }

                    double pyloricness;
                    pyloricness = PyloricPerformance(Circuit); //HP is left on during test

                    testfile << pyloricness << " ";
                }
                testfile << endl;
            }
            testfile.close();
        }
    }

    // ONE HP MODE
    // ifstream evolvedHP;
    // ifstream ifs;
    // char HPfname[] = "./Generalist HP Mechanisms/100/bestind.dat"; 
    // evolvedHP.open(HPfname);
    // if (!evolvedHP) {
    //     cerr << "File not found: " << HPfname << endl;
    //     exit(EXIT_FAILURE);
    // }

     //ONE HP MODE
    // Circuit.SetHPPhenotype(evolvedHP,StepSize,true);
    // cout << Circuit.l_boundary << " " << Circuit.u_boundary << endl;

    // bool all_pyloric = true; //is true until proven false for one initial condition
    

    // char fname[] = "../Pyloric CTRNN Genomes/goodgenomesnormalizedfit.dat";
    // ifs.open(fname);
    // if (!ifs) {
    //     cerr << "File not found: " << fname << endl;
    //     exit(EXIT_FAILURE);
    // }

    // TVector<double>phenotype(1,CTRNNphenotypelen);
    // double pyl_fitness =  0;
    // for (int indiv=0;indiv<num_indivs;indiv++){

    //     ifs >> phenotype;
    //     ifs >> pyl_fitness;
        
    //     phenotype >> Circuit;
    //     // cout << Circuit.biases << endl;

    //     for (int j = 0;j<1;j++){ //5
    //         // cout << "j" << j << endl;
    //         // char indiv_char[Max_Digits + sizeof(char)];
    //         // std::sprintf(indiv_char, "%d", indiv);
                
    //         // char infile[(2*Max_Digits) + (2*sizeof(char)) + 51+13 ];
    //         // strcpy(infile, "./Specifically Evolved HP mechanisms/Every Circuit/");
    //         // strcat(infile, indiv_char);
    //         // strcat(infile, "/");

    //         // char evol_char[1 + sizeof(char)];
    //         // std::sprintf(evol_char, "%d", j);

    //         // strcat(infile,evol_char);
    //         // strcat(infile, "/bestind.dat");

    //         // evolvedHP.open(infile);
    //         // if (!evolvedHP) {
    //         //     cerr << "File not found: " << infile << endl;
    //         //     exit(EXIT_FAILURE);
    //         // }

    //         // evolvedHP.open("./Specifically Evolved HP mechanisms/Every Circuit/0/0/bestind.dat");
    //         // TVector<int> plastpars(1,N+(N*N));
    //         // evolvedHP >> plastpars;
    //         // TVector<double> HPgenotype(1,(num*4));
    //         // evolvedHP >> HPgenotype;
    //         // TVector<double> HPphenotype(1,(num*4));
    //         // evolvedHP >> HPphenotype;
    //         // Circuit.SetHPPhenotype(HPphenotype,StepSize,true);


    //         //IN THE BIAS1,BIAS3 PLANE
    //         for (int i=1;i<=num_pts;i++){
    //             Circuit.SetNeuronBias(1,ptlist(i,1));
    //             Circuit.SetNeuronBias(3,ptlist(i,2));
    //             Circuit.RandomizeCircuitOutput(0.5, 0.5);

    //             // Run the circuit for an initial transient; HP is off and fitness is not evaluated
    //             for (double t = StepSize; t <= TransientDuration; t += StepSize) {
    //                 Circuit.EulerStep(StepSize,false);
    //             }
    //             // cout << "transient done" << endl;

    //             for (double t = StepSize; t <= PlasticDuration; t += StepSize) {
    //                 Circuit.EulerStep(StepSize,true);
    //             }

    //             // cout << "plasticity period done" << endl;

    //             double pyloricness;
    //             pyloricness = PyloricPerformance(Circuit); //HP is left on during test

    //             // cout <<"Pyloricness tested" << endl;
    //             testfile << pyloricness << " ";
    //             // if (pyloricness < .3){
    //                 // all_pyloric = false;
    //             // }
    //         }
    //         //INDIVIDUALIZED HP MODE
    //         testfile << endl;
    //         evolvedHP.close();
    //     }
    //     testfile << endl;
    //     // testfile << all_pyloric << endl;
    // }

    // //ONE HP MODE
    // evolvedHP.close();

    // ifs.close();
    // testfile.close();

    return 0;
}