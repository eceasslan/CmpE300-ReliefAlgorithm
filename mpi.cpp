/*
Student Name: Ece Dilara Aslan
Compile Status: Compiling
Program Status: Working
Notes: This source file uses Open MPI version 4.0.5 and it is compiled with Apple clang version 11.0.3 (clang-1103.0.32.62).
*/

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>

using namespace std;

int p; //number of processors
int n, a, m, t; //number of instances, number of features, number of iterations in Relief algorithm, number of top weighted features per slave processor
int instance; //number of instances per slave processor

//finds maximum and minimum value of each feature for current processor
void findMaxMin(long double array[], long double max[], long double min[]){
    copy(array, array+a, max);
    copy(array, array+a, min);
    for(int i=1; i<instance; i++){ //iterates through instances
        for(int j=0; j<a; j++){ //iterates through features
            if(array[(i*(a+1))+j] > max[j]){
                max[j] = array[(i*(a+1))+j];
            }
            else if(array[(i*(a+1))+j] < min[j]){
                min[j] = array[(i*(a+1))+j];
            }
        }
    }
    return;
}

//calculates the Manhattan distance between inst1 and inst2
long double manhattanDistance(long double inst1[], long double inst2[]){
    long double sum = 0;
    for(int i=0; i<a; i++){
        sum = sum + abs(inst1[i]-inst2[i]);
    }
    return sum;
}

//finds nearest hit and miss of current target instance
void findHitMiss(long double array[], long double targetInstance[], int target, long double hit[], long double miss[]){
    long double hitDist = -1; //stores the distance between current nearest hit and target instance
    long double missDist = -1; //stores the distance between current nearest miss and target instance
    for(int i=0; i<instance; i++){ //iterates through instances
        if(i != target){ //calculates Manhattan distance between the target instance and another instance
            long double temp[a+1]; //stores feature values and class label of current instance
            for(int j=0; j<=a; j++){
                temp[j] = array[(i*(a+1))+j];
            }
            long double distTemp = manhattanDistance(targetInstance, temp); //calculates and stores Manhattan distance between current instance and target instance
            if(temp[a] == targetInstance[a]){ //if it is a hit, checks whether it is the nearest
                if(hitDist == -1 || distTemp < hitDist){
                    copy(temp, temp+a, hit);
                    hitDist = distTemp;
                }
            }
            else{ //if it is a miss, checks whether it is the nearest
                if(missDist == -1 || distTemp < missDist){
                    copy(temp, temp+a, miss);
                    missDist = distTemp;
                }
            }
        }
    }
    return;
}

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv); //initializes MPI environment
    
    int rank; //rank of the processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream file;
    file.open(argv[1], ios::in);
    file >> p; 
    file >> n; 
    file >> a; 
    instance = n/(p-1); //calculates the number of instances per slave processor
    if(rank != 0){ //if the processor is not the master, work of the file stream is done, so closes the file stream
        file.close();
    }

    long double features[(n+instance)*(a+1)]; //stores all of the instance data and an unused space at the beginning
    long double subfeatures[instance*(a+1)]; //stores a subset of the instance data associated with the current processor
    int topFeatures[t*p]; //stores ids of all top weighted features founded by the slave processors and an unused space at the beginning
    int subTopFeatures[t]; //stores ids of the top weighted features founded by the current processor
    
    if(rank == 0){ //master processor
        file >> m; 
        file >> t; 
        for(int i=instance; i<n+instance; i++){ //gets instance data from the file
            for(int j=0; j<=a; j++){
                file >> features[(i*(a+1))+j];
            }
        }
        file.close(); //work of the file stream is done, so closes the file stream
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(features, instance*(a+1), MPI_LONG_DOUBLE, subfeatures, instance*(a+1), MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);
    //scatters the instance data between the slave processors

    if(rank != 0){ //slave processors

        long double weights[a]; //stores the weights of the features
        long double max[a]; //stores maximum value of each feature
        long double min[a]; //stores minimum value of each feature
        fill(weights, weights+a, 0.0); //initializes all weights to 0
        findMaxMin(subfeatures,max,min); //finds maximum and minimum value of each feature

        for(int i=0; i<m; i++){ //weight calculation iterations

            long double targetInstance[a+1]; //stores feature values and class label of target instance
            for(int j=0; j<=a; j++){
                targetInstance[j] = subfeatures[(i*(a+1))+j];
            }

            long double hit[a]; //stores feature values of the nearest hit
            long double miss[a]; //stores feature values of the nearest miss
            findHitMiss(subfeatures, targetInstance, i, hit, miss); //finds nearest hit and miss

            for(int j=0; j<a; j++){ //weight calculation according to diff function
                long double diffHit;
                long double diffMiss;
                diffHit = abs(targetInstance[j]-hit[j])/(max[j]-min[j]);
                diffMiss = abs(targetInstance[j]-miss[j])/(max[j]-min[j]);
                weights[j] = weights[j] - (diffHit/(long double)m) + (diffMiss/(long double)m);
            }
        }

        long double* ptr = weights; //a pointer to the beginning of the weights array
        for(int i=0; i<t; i++){ //finds the top weighted features
            long double* maxItr = max_element(weights, weights+a);
            subTopFeatures[i] = maxItr-ptr;
            *maxItr = 2.22507e-308;
        }
        sort(subTopFeatures, subTopFeatures+t); //sorts top weighted features according to their ids
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    //writes top weighted features
    if(rank != 0){ //slave processors
        cout << "Slave P" << rank << " :"; 
        for(int i=0; i<t; i++){
            cout << " " << subTopFeatures[i];
        }
        cout << endl;
    }

    MPI_Gather(subTopFeatures, t, MPI_INT, topFeatures, t, MPI_INT, 0, MPI_COMM_WORLD);
    //gathers top weighted features from all of the slave processors

    //writes top weighted features
    if(rank == 0){ //master processor
        this_thread::sleep_for(chrono::seconds(1));
        sort(topFeatures+t, topFeatures+(t*p)); //sorts the top weighted features according to their ids
        int tempFeature = -1; //stores the current feature id
        cout << "Master P0 :";
        for(int i=t; i<t*p; i++){
            if(tempFeature != topFeatures[i]){
                cout << " " << topFeatures[i];
                tempFeature = topFeatures[i];
            }
        }
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize(); //finishes MPI environment
    return 0;
}