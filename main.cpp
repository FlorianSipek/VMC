/*
 * main.cpp
 *
 *  Created on: Jul 13, 2018
 *      Author: sipek
 */



#include <iostream> // Cout
#include <iomanip> //Manipulators
#include <vector>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <limits>
#include <functional>
#include <random>
#include <array>
//#include <omp.h>
#include <cstdio>

//#include "Particles.h"

#include<Eigen/Dense>

using namespace std;
using namespace Eigen;

//double L = 1;
mt19937_64 gen;

unsigned N = 343, dim = 3, steps = 1e6;
double rho;
double L;		// Length calculation
double d;		// lattice constant for lattice init
double rc1, rc2;
unsigned mode = 1; 		// mode 1 = jastrow 1


double dice_real(double L){
	uniform_real_distribution<double> distr(-L/2.0, L/2.0);
	return distr(gen);
}

double dice_int(int N){
//	random_device gen;
	std::uniform_int_distribution<int> distrint(0,N);
	return distrint(gen);
}

double GaussDistr(double x, double mu){
	normal_distribution<double> gauss(x,mu);
	return gauss(gen);
}

VectorXd smooth_cutoff(double a){
	VectorXd b(6);
	b.setZero();
	MatrixXd A(6,6);
	A.setZero();


	A(0,0) = pow(rc1,5);
	A(0,1) = pow(rc1,4);
	A(0,2) = pow(rc1,3);
	A(0,3) = pow(rc1,2);
	A(0,4) = rc1;
	A(0,5) = 1;

	A(1,0) = 5*pow(rc1,4);
	A(1,1) = 4*pow(rc1,3);
	A(1,2) = 3*pow(rc1,2);
	A(1,3) = 2*rc1;
	A(1,4) = 1;


	A(2,0) = 20*pow(rc1,3);
	A(2,1) = 12*pow(rc1,2);
	A(2,2) = 6*rc1;
	A(2,3) = 2;



	A(3,0) = pow(rc2,5);
	A(3,1) = pow(rc2,4);
	A(3,2) = pow(rc2,3);
	A(3,3) = pow(rc2,2);
	A(3,4) = rc2;
	A(3,5) = 1;

	A(4,0) = 5*pow(rc2,4);
	A(4,1) = 4*pow(rc2,3);
	A(4,2) = 3*pow(rc2,2);
	A(4,3) = 2*rc2;
	A(4,4) = 1;


	A(5,0) = 20*pow(rc2,3);
	A(5,1) = 12*pow(rc2,2);
	A(5,2) = 6*rc2;
	A(5,3) = 2;


	b[0] = (1-a/rc1)/(1-a/rc2);
	b[1] = (a/pow(rc1,2))*(1/(1-a/rc2));
	b[2] = -2*(a/pow(rc1,3))*(1/(1-a/rc2));
	b[3] = 1;
//	cout << "Matrix: " << A << endl << endl;
//	cout << "right side: " << b << endl << endl;

//	VectorXd out = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
	VectorXd out = A.colPivHouseholderQr().solve(b);

//	cout << "output vector = " << out << endl;

	return out;
}

VectorXd cutoff(6);



class Particles{
	public:
	unsigned N, DIM;
	double L;
	vector<VectorXd> coordinates, coordinatesNIC;
//	void random_init(unsigned,unsigned);
	Particles(unsigned, unsigned, double);	//random init
	Particles(const Particles&);			//copy constructor
	Particles(unsigned, string, double);	//init from coordinates file
	Particles(unsigned, unsigned, double, double);		//lattice init
	void print_coordinates(string);
	void print_coordinatesNIC(string);
	VectorXd get_coordinates(unsigned);
	double get_coordinatesNIC(double);
	unsigned shift_randomly(double);
	VectorXd distance(unsigned i, unsigned j);
	VectorXd distanceNIC(unsigned i, unsigned j);
};
Particles::Particles(unsigned n, unsigned d, double l){	//Constructor for random particle initialization
	N = n;
	DIM = d;
	L = l;
	int k;
	double r;
	coordinates.resize(N);
	coordinatesNIC.resize(N);
	for(unsigned i = 0; i < N; i++){
		coordinates[i].resize(DIM);
		coordinatesNIC[i].resize(DIM);
	}
	for(unsigned i = 0; i < N; i++){
		for(unsigned j = 0; j < DIM;j++){
			coordinates[i][j] = dice_real(L);
			r = coordinates[i][j];
			k = r/L + ((r >= 0.0) ? 0.5 : -0.5);
			coordinatesNIC[i][j] = r - k*L;
		}
	}
}
Particles::Particles(const Particles &P2){
	coordinates = P2.coordinates;
	coordinatesNIC = P2.coordinatesNIC;
	N = P2.N;
	L = P2.L;
	DIM = P2.DIM;
}

Particles::Particles(unsigned d, string in, double l){
	DIM = d;
	L = l;
	ifstream input;
	double x;
	VectorXd row(3);
	unsigned n = 0;

	input.open(in);
	if (!input.is_open()){
		cerr << "Unable to open file" << "\n";
		exit(-1);
	}

	while (input && !input.eof()){
		for(unsigned i = 0; i < DIM; i++){
			input >> x;
			row[i] = x;
		}
		coordinates.push_back(row);
		n++;
	}
	input.close();
	N = n - 1;
	coordinates.resize(N);

	int k;
	double r;
	coordinatesNIC.resize(N);
	for(unsigned i = 0; i < N; i++){
		coordinatesNIC[i].resize(DIM);
	}
	for(unsigned i = 0; i < N; i++){
		for(unsigned j = 0; j < DIM;j++){
			r = coordinates[i][j];
			k = (r/L) + ((r >= 0.0) ? 0.5 : -0.5);
			coordinatesNIC[i][j] = r - k*L;
		}
	}
}

Particles::Particles(unsigned n, unsigned dim, double d, double l){
	N = n;
	L = l;
	DIM = dim;
	int N1D = (int)cbrt(N);

	coordinates.resize(N);
	coordinatesNIC.resize(N);
	for(unsigned i = 0; i < N; i++){
		coordinates[i].resize(DIM);
		coordinatesNIC[i].resize(DIM);
	}

	for(int i = 0; i < N1D; i++){
		for(int j = 0; j < N1D; j++){
			for(int k = 0; k < N1D; k++){
					coordinates[k+N1D*j+N1D*N1D*i][0] = k*d - L/2 + d/2;
					coordinates[k+N1D*j+N1D*N1D*i][1] = j*d - L/2 + d/2;
					coordinates[k+N1D*j+N1D*N1D*i][2] = i*d - L/2 + d/2;
			}
		}
	}
	coordinatesNIC = coordinates;
}

void Particles::print_coordinates(string name){			//prints all particle coordinates to console
	ofstream file;
	file.open(name, std::ofstream::out | std::ofstream::app);
	for(unsigned i = 0; i < N; i ++){
		for(unsigned j = 0; j < DIM; j++){
			file << coordinates[i][j] << "\t";
		}
		file << endl;
	}
	file.close();
}
void Particles::print_coordinatesNIC(string name){			//prints all particle coordinates to console
	ofstream file;
	file.open(name, std::ofstream::out | std::ofstream::app);
	for(unsigned i = 0; i < N; i ++){
		for(unsigned j = 0; j < DIM; j++){
			file << coordinatesNIC[i][j] << "\t";
		}
		file << endl;
	}
	file.close();
}

/*
void Particles::print_coordinatesNIC(){			//prints all particle coordinates to console
	for(unsigned i = 0; i < N; i ++){
		for(unsigned j = 0; j < DIM; j++){
			cout << coordinatesNIC[i][j] << "\t";
		}
		cout << endl;
	}
}
*/
VectorXd Particles::get_coordinates(unsigned n){		//returns coordinates of particle n
	return coordinates[n];
}

double Particles::get_coordinatesNIC(double r){
//	VectorXd out(3);
//	out[0] = coordinates[n][0] - ((int) coordinates[n][0]*(1/L)+((coordinates[n][0] >= 0.0) ? 0.5 : -0.5))*L;
//	out[1] = coordinates[n][1] - ((int) coordinates[n][1]*(1/L)+((coordinates[n][1] >= 0.0) ? 0.5 : -0.5))*L;
//	out[2] = coordinates[n][2] - ((int) coordinates[n][2]*(1/L)+((coordinates[n][2] >= 0.0) ? 0.5 : -0.5))*L;
	int k;
    k =  r / L + ((r >= 0.0) ? 0.5 : -0.5);
	return r - k*L;
}

unsigned Particles::shift_randomly(double a){
//	uniform_int_distribution<int> distrint(0,N);
	unsigned n = dice_int(N-1);
	double rtrho = cbrt(rho);
//	cout << n << endl << endl;
	for(unsigned j = 0; j < DIM; j++){
//		coordinates[n][j] += dice_real(L);
		coordinates[n][j] += (1./rtrho)*GaussDistr(0,1);
		coordinatesNIC[n][j] = get_coordinatesNIC(coordinates[n][j]);
	}
	return n;
}

VectorXd Particles::distance(unsigned i, unsigned j){
	VectorXd delta(3);
	delta = coordinates[i] - coordinates[j];
	return delta;
}

VectorXd Particles::distanceNIC(unsigned i, unsigned j){
	VectorXd dr(3);
	VectorXd delta(3);
	delta = coordinates[i] - coordinates[j];
	for(unsigned n = 0; n < DIM; n++){
		dr[n] = get_coordinatesNIC(delta[n]);
	}
	if(dr.norm()>((L/2.0)*sqrt(3.0)))
		cout << "Error!" << endl << dr.norm() << endl;
	return dr;
}

class Jastrow{
	public:
	double jastrow;
	Jastrow(VectorXd, double);
	Jastrow(VectorXd,double,double);

//	VectorXd smooth_cutoff(double,double,double);
};


Jastrow::Jastrow(VectorXd dr, double a){
	double r = dr.norm();

	if((dr).norm() <= a){
		jastrow = 0;
	}
	else if(rc1 < dr.norm() && dr.norm() <= rc2){
		jastrow = cutoff[0]*pow(r,5) + cutoff[1]*pow(r,4) + cutoff[2]*pow(r,3) + cutoff[3]*pow(r,2) + cutoff[4]*r + cutoff[5];
	}
	else if((dr).norm() > a && dr.norm() <= rc1){
		jastrow = (1-(a/r))/(1-a/rc2);
	}
	else {
		jastrow = 1.;
	}
}
Jastrow::Jastrow(VectorXd dr, double a, double alpha){
	if((dr).norm() <= a){
		jastrow = 0;
	}
	else if((dr).norm() > a){
		jastrow = 1-exp(-((dr).norm() - a)/alpha);
	}
}


/*
double Jastrow1(VectorXd x, VectorXd y){
	if((x-y).norm() <= a){
		return 0;
	}
	else if((x-y).norm() > a){
		return 1-(a/(x-y).norm());
	}
}

double Jastrow2(VectorXd x, VectorXd y, double alpha){
		if((x-y).norm() <= a){
		return 0;
	}
	else if((x-y).norm() > a){
		return 1-exp(-((x-y).norm() - a)/alpha);
	}
}
*/
void write_to_data(string name, double x, double y){
	ofstream file;
	file.open(name, std::ofstream::out | std::ofstream::app);
	file << x << "\t" << y << endl;
	file.close();

//	FILE *fp;
//	fp = fopen("f1.dat","a");
//	fprintf(fp, "%f \t %f\n", x, y);
//	fclose(fp);
}

void write_to_3Ddata(string name, double x, double y, double yerr){
	ofstream file;
	file.open(name, std::ofstream::out | std::ofstream::app);
	file << x << "\t" << y << "\t" << yerr << endl;
	file.close();
}

//double WaveFct(Particles boson, double a, double alpha){
//	double out = 1;
//	for(unsigned j = 0; j < boson.N; j++){
//		for(unsigned i = 0; i < j; i++){
////			Jastrow factor(boson.distanceNIC(i,j),a);
//			Jastrow factor(boson.distanceNIC(i,j),a,alpha);
////			out *= (1+a)*factor.jastrow;
//			out *= factor.jastrow;
//		}
//	}
//	return out;
//}

class calc_wavefct{
public:
	double wavefunction, A, Alpha;
	calc_wavefct(double, double);
	double full_wavefct(Particles&);
	double update_wavefct(Particles&, unsigned);
};

calc_wavefct::calc_wavefct(double a, double alpha){
	A = a;
	Alpha = alpha;
	wavefunction = 1;
}

double calc_wavefct::full_wavefct(Particles &boson){
	double out = 1;
	for(unsigned j = 0; j < boson.N; j++){
		for(unsigned i = 0; i < j; i++){
			if(mode == 1){
				Jastrow factor(boson.distanceNIC(i,j),A);
				out *= (1+A)*factor.jastrow;
			}
			else{
				Jastrow factor(boson.distanceNIC(i,j),A,Alpha);
				out *= factor.jastrow;
			}
		}
	}
	wavefunction = out;
	return out;
}

double calc_wavefct::update_wavefct(Particles &boson, unsigned n){
	double out = 1;
	for(unsigned i = 0; i < boson.N; i++){
		if(i != n){
			if(mode == 1){
//				out*= 1+A;
				if((boson.distanceNIC(i,n).norm()) <= ((boson.L)/2)){
					Jastrow factor(boson.distanceNIC(i,n),A);
					out *= factor.jastrow;
				}
			}
			else{
				if((boson.distanceNIC(i,n).norm()) <= ((boson.L)/2)){
					Jastrow factor(boson.distanceNIC(i,n),A,Alpha);
					out *= factor.jastrow;
				}
			}
		}
	}
	wavefunction = out;
	return out;
}

//double localEnergy1(Particles &boson, double a){
//	double out = 0;
//	VectorXd term1(3);
//	term1.setZero();
//	double term2 = 0;
//	double term3 = 0;
//	for(unsigned k = 0; k < boson.N; k++){
//		for(unsigned j = 0; j < boson.N; j++){
//			if(j != k){
////				term1 += (a/((pow((boson.distanceNIC(k,j)).norm(),2))*((boson.distanceNIC(k,j).norm())-a)))*boson.distanceNIC(k,j);
////				term2 += a*(((boson.distanceNIC(k,j)).norm()-a-pow((boson.distanceNIC(k,j)).norm(),2))/(pow((boson.distanceNIC(k,j)).norm(),2)*pow((boson.distanceNIC(k,j)).norm()-a,2)));
//				term1 += (a/(1-(a/boson.distanceNIC(k,j).norm())))*(1/pow(boson.distanceNIC(k,j).norm(),3))*boson.distanceNIC(k,j);
//				term2 += (a*a/(pow(1-(a/boson.distanceNIC(k,j).norm()),2)))*(1/(pow(boson.distanceNIC(k,j).norm(),3)));
//				term3 += 3*((1/(pow(boson.distanceNIC(k,j).norm(),3))) - (1/(pow(boson.distanceNIC(k,j).norm(),2))))*(a/(1-(a/(boson.distanceNIC(k,j).norm()))));
//			}
//		}
//		out += term1.dot(term1) - term2 + term3;
//	}
//
//	return out;
//}

//double localEnergy2(Particles &boson, double a, double alpha){
//	double out = 0;
//	VectorXd term1(3);
//	term1.setZero();
//	double term2 = 0, term3 = 0;
//	for(unsigned k = 0; k < boson.N; k++){
//		for(unsigned j = 0; j < boson.N; j++){
//			if(j != k){
//				term1 += (1/alpha)*(exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)/(1-exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)))*(boson.distanceNIC(k,j)/(boson.distanceNIC(k,j).norm()));
//				term2 += (1/alpha*alpha)*(exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)/(1-exp(-(boson.distanceNIC(k,j).norm()-a)/alpha))) + (exp(-2*(boson.distanceNIC(k,j).norm()-a)/alpha)/pow(1-exp(-(boson.distanceNIC(k,j).norm()-a)/alpha),2));
//				term3 += (1/alpha)*(2/boson.distanceNIC(k,j).norm())*(exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)/(1-exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)));
//			}
//		}
//		out += term1.dot(term1) - term2 + term3;
//	}
//
//	return out;
//}

class localEnergy{
public:
	double Eloc;
	localEnergy(Particles&, double);
	localEnergy(Particles&, double, double);
};

localEnergy::localEnergy(Particles &boson, double a){
	double out = 0;
	VectorXd term1(3);
	term1.setZero();
	double term2 = 0;
	double term3 = 0;
	for(unsigned k = 0; k < boson.N; k++){
		term1.setZero();
		term2 = 0;
		term3 = 0;
		for(unsigned j = 0; j < boson.N; j++){
			if(j != k){
//				term1 += (a/((pow((boson.distanceNIC(k,j)).norm(),2))*((boson.distanceNIC(k,j).norm())-a)))*boson.distanceNIC(k,j);
//				term2 += a*(((boson.distanceNIC(k,j)).norm()-a-pow((boson.distanceNIC(k,j)).norm(),2))/(pow((boson.distanceNIC(k,j)).norm(),2)*pow((boson.distanceNIC(k,j)).norm()-a,2)));
				//term1 += (a/(1-(a/boson.distanceNIC(k,j).norm())))*(1/pow(boson.distanceNIC(k,j).norm(),3))*boson.distanceNIC(k,j);
				//term2 += (a*a/(pow(1-(a/boson.distanceNIC(k,j).norm()),2)))*(1/(pow(boson.distanceNIC(k,j).norm(),4)));
				VectorXd r = boson.distanceNIC(k,j);
				double rkj = boson.distanceNIC(k, j).norm();
				if (rkj < L/2.0)
				{
					if(rkj < rc1){
						double a_rkj = a / rkj;
						double one_a_rkj = 1 - a_rkj;
						double rkj3 = rkj * rkj * rkj;
						double rkj4 = rkj3 * rkj;
						VectorXd tmp1 = ((a / one_a_rkj) / rkj3) * boson.distanceNIC(k, j);
						double tmp2 = ((a * a) / (one_a_rkj * one_a_rkj)) / rkj4;
						//cout << "term1=" << tmp1.dot(tmp1) << " term2=" << tmp2 << endl;
						term1 += tmp1;
						term2 += tmp2;
					}
					if(rkj >=rc1 && rkj <= rc2){
						// Jastrow smooth(r,a);
						// VectorXd cutoff = smooth.smooth_cutoff(a,rc1,rc2);
						double f1 = 0;
						double f2 = 0;
						double f4 = 0;
						for(unsigned l = 0; l < 6; l++){
							f1 += cutoff[5-l]*pow(rkj,l);
						}
						for(unsigned l = 1; l < 6; l++){
							f2 += cutoff[5-l]*l*pow(rkj,l-1);
//							f3 += cutoff[5-l]*l*pow(rkj,l-2);
//							f4 += l*(l+1)*cutoff[5-l]*pow(rkj,l-2);
						}
						for(unsigned l = 2; l < 6; l++){
							f4 += l*(l-1)*cutoff[5-l]*pow(rkj,l-2);
						}
						term1 += (1/f1)*f2*(r/rkj);
						term2 += f2*(pow(f1,-2))*f2;
						term3 += (1/f1)*(f4);
					}
				}
				//term3 += 3*((1/(pow(boson.distanceNIC(k,j).norm(),3))) - (1/(pow(boson.distanceNIC(k,j).norm(),2))))*(a/(1-(a/(boson.distanceNIC(k,j).norm()))));
			}
		}
		out += -term1.dot(term1) + term2 - term3;
	}
	Eloc = out;
}

localEnergy::localEnergy(Particles &boson, double a, double alpha){
	double out = 0;
	VectorXd term1(3);
	term1.setZero();
	double term2 = 0, term3 = 0;
	for(unsigned k = 0; k < boson.N; k++){
		term1.setZero();
		term2 = 0;
		term3 = 0;
		for(unsigned j = 0; j < boson.N; j++){
			if(j != k){
				double rkj = boson.distanceNIC(k,j).norm();
				VectorXd r = boson.distanceNIC(k,j);
				double efct = exp(-(rkj-a)/alpha);
				double efctone = 1 - efct;
				if(rkj < L/2.0){
					// term1 += (1/alpha)*(exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)/(1-exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)))*(boson.distanceNIC(k,j)/(boson.distanceNIC(k,j).norm()));
					// term2 += (1/(alpha*alpha))*(exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)/(1-exp(-(boson.distanceNIC(k,j).norm()-a)/alpha))) + (exp(-2*(boson.distanceNIC(k,j).norm()-a)/alpha)/pow(1-exp(-(boson.distanceNIC(k,j).norm()-a)/alpha),2));
					// term3 += (1/alpha)*(2/boson.distanceNIC(k,j).norm())*(exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)/(1-exp(-(boson.distanceNIC(k,j).norm()-a)/alpha)));
					term1 += (1/alpha)*1/(efctone)*efct*(1/rkj)*r;
					term2 += (1/(alpha*alpha))*((1/(efctone*efctone))*efct*efct + (efct/efctone));
					term3 += (2/(rkj*alpha*efctone))*efct;
				}
			}
		}
		out += -term1.dot(term1) + term2 - term3;
	}
	Eloc = out;
}
VectorXd histogrammean(100);


void debug_histogram(Particles &boson){

	unsigned index = 0, noofbins = 100;
//	double binsize = (boson.L/2)/noofbins;
	VectorXi histogram(noofbins);
	histogram.setZero();
	double distance = 0;
	for(unsigned i = 0; i < boson.N; i++){
		for(unsigned j = 0; j < i; j++){
			distance = boson.distanceNIC(i,j).norm();
			if(distance < boson.L/2){
				index = (int)((distance/(boson.L/2))*noofbins);
				histogram[index]++;
			}
		}
	}
	for(unsigned n = 0; n < histogram.size(); n++){
		histogrammean[n]+=histogram[n]/(pow((n+1),2));
//		write_to_data("histogram_1.dat",n,histogram[n]/pow((n+1),2));
	}

}

void Metropolis(Particles &boson, double a, unsigned steps, double alpha, double rho){
	int accept = 0, p = 0, m = 250;
	double psiold, psinew;
	Particles shifted = boson;


	calc_wavefct wavefct(a, alpha);


	double E = 0;
	double esq = 0, Eloc = 0;
//	double E = localEnergy1(boson,a);
//	double E = localEnergy(boson,a,alpha).Eloc;

	for(unsigned n = 0; n < steps; n++){
//		psiold = WaveFct(boson,a,alpha);
		unsigned r = shifted.shift_randomly(a);
//		psiold = wavefct.full_wavefct(boson);
//		psinew = wavefct.full_wavefct(shifted);
//		cout << "ratio 1 = " << (pow(psinew,2)/pow(psiold,2)) << endl;
		psiold = wavefct.update_wavefct(boson,r);
		psinew = wavefct.update_wavefct(shifted,r);
//		cout << "ratio 2 = " << (pow(psinew,2)/pow(psiold,2)) << endl;
//		psinew = WaveFct(shifted, a, alpha);

//		cout << "psiold " << psiold << endl;
//		cout << "psinew " << psinew << endl;
//		cout << "ratio " << (pow(psinew,2)/pow(psiold,2)) << endl;
		double dice = (dice_real(1)+0.5);
//		cout << "random number " << dice << endl;

//		Eloc = localEnergy(boson, a).Eloc;

		if(dice < (pow(psinew,2)/pow(psiold,2))){
			accept++;
			boson.coordinates = shifted.coordinates;
			boson.coordinatesNIC = shifted.coordinatesNIC;
		}

		else{
			shifted.coordinates = boson.coordinates;
			shifted.coordinatesNIC = boson.coordinatesNIC;
		}
		if(n%m == 0 && n > 500){
			if(mode == 1){
				Eloc = localEnergy(boson, a).Eloc;
			}
			else if (mode == 2){
				Eloc = localEnergy(boson,a,alpha).Eloc;
			}
			else{
				cerr << "ERROR! Mode not known!" << endl;
				break;
			}
			E += Eloc;
			esq += Eloc*Eloc;
//			E += localEnergy2(boson,a,alpha);
//			cout << "Energy = " << E << endl;
			p++;
//			write_to_data("Energy_" + to_string(mode) + ".dat",(double)p, (Eloc));
//			debug_histogram(boson);

//			if(p==822){
//				string name = "coordinates" + to_string(n);
//
//				boson.print_coordinates(name+".dat");
//				boson.print_coordinatesNIC(name+"NIC.dat");
//			}
		}

	cout << "Acceptance ratio: " << (accept/(n+1.0))*100  << endl << endl;

//	cout << "Wave function: " << psinew << endl;
//	write_to_data("MC_2.dat",(double)n, psiold);
	}
	double error = sqrt(abs((1/p)*esq - pow((E/p),2)))/sqrt(p);
//	double error = sqrt(abs((1/p)*esq - pow((E/p),2)));
//	write_to_3Ddata("final_output_" + to_string(mode) + ".dat",a*rho,(E/p)/N,error/N);
//	debug_histogram(boson);
//	write_to_3Ddata("f_2-optimize.dat",alpha,(E/p)/N,error/N);
	write_to_3Ddata("f_2_alpha_rho.dat",alpha,a*rho,(E/p)/N);
//	return E/p;
}



int main() {
	histogrammean.setZero();

//	boson.print_coordinates("Lattice.dat");
//	Metropolis(boson,a,steps);
//	boson.print_coordinates("Coordinates.dat");
//	boson.print_coordinatesNIC("CoordinatesNIC.dat");
//	cout << "Last energy = " << Metropolis(boson,a,steps) << endl << endl;
//	cout << "a*rho^1/3 = " << a*cbrt(rho) << endl << endl;
	double epsa = 1e-2;

	double alpha = 0.1;
	double a = 0.01;
//	cutoff.setZero();

	rho = 1.0;
	L = cbrt(N/rho);		// Length calculation
	d = cbrt(1/rho);		// lattice constant for lattice init
	rc1 = 0.7*(L/2) , rc2 = L/2;

	cutoff = smooth_cutoff(a);


//	for(unsigned i = 0; i < 6; i++){
//		write_to_data("cutoff.dat",i,cutoff[i]);
//	}
//
//	cout << "L = " << L << endl << "rho = " << rho << endl << "a = " << a << endl;
//
//	Particles boson(N,dim,d,L);
//	boson.print_coordinates("Coordinates_L=" + to_string(L) +".dat");


//#pragma omp parallel for private(a)
// 	for(unsigned i = 0; i < 1000; i++){
// //		a = (i+1)*epsa;
// 		if(a*rho > 1.)
// 			continue;
// 		if(mode == 1)
// 			cutoff = smooth_cutoff(a);
//
// 		Particles boson(N,dim,d,L);		// Coordinate initialization with crystal structure
// 		Metropolis(boson,a,steps,alpha,rho);
// //		write_to_data("out_1.dat",a*rho,Metropolis(boson,a,steps,alpha));
// //		write_to_data("out_2.dat",a*rho,Metropolis(boson,a,steps,alpha));
// //		rho += epsr;
// 		a += epsa;
// 	}

/******* Single run *******/
// 		Particles boson(N,dim,d,L);		// Coordinate initialization with crystal structure
////		Particles boson(dim,"coordinates411500.dat",L);
// 		Metropolis(boson,a,steps,alpha,rho);
//
// 		for(unsigned k=0;k<100;k++){
// 			write_to_data("histogram_" + to_string(mode) + ".dat",k,histogrammean[k]);
// 		}

/******* ALPHA optimization *******/
//	#pragma omp parallel for private(alpha)
//	for(unsigned j = 0; j < 50; j++){
////		alpha = (j+1)*epsa;
//		 if(alpha > 0.2)
//		 	continue;
//
//		Particles boson(N,dim,d,L);
//		Metropolis(boson,a,steps,alpha,rho);
//		alpha += epsa;
//	}

 /******* ALPHA optimization, 3D *******/
	mode = 2;
	 	while(a*rho < 0.3){
	 //		a = (i+1)*epsa;
	 		alpha = 0.3;
	 		Particles boson(N,dim,d,L);		// Coordinate initialization with crystal structure
	 			while(alpha <= 0.5){
	 		//		alpha = (j+1)*epsa;

	 				Metropolis(boson,a,steps,alpha,rho);
	 				alpha += epsa;
	 			}
	 //		write_to_data("out_1.dat",a*rho,Metropolis(boson,a,steps,alpha));
	 //		write_to_data("out_2.dat",a*rho,Metropolis(boson,a,steps,alpha));
	 //		rho += epsr;
	 		a += epsa;
	 	}



/****** JASTROW TEST ******/
//	VectorXd test(3);
//	test.setZero();
//
//	while(test.norm()<(L/2)){
//		Jastrow func(test,a,alpha);
//		write_to_data("f2.dat", test.norm(), func.jastrow);
//		test[0] += epsa;
//	}


	/*
	Particles bosons(N, dim, L);

	cout << endl << endl;
	Metropolis(bosons,a,steps);
	bosons.print_coordinates("InitCoordinates.dat");
	bosons.print_coordinatesNIC("InitCoordinatesNIC.dat");
*/

//	Particles boson(dim,"InitCoordinates.dat",L);
//	boson.print_coordinates("test.dat");
//	boson.print_coordinatesNIC("testNIC.dat");
//	cout << "N = " << boson.N << endl;

//	Metropolis(boson,a,steps);

//	VectorXd tmp = atoms.get_coordinates(0);
//	for(unsigned i = 0; i < tmp.size();i++){
//		cout << tmp[i] << "\t";
//	}
/*
	cout << endl << endl;
	atoms.shift_randomly();
	atoms.print_coordinates();
	cout << endl;
	atoms.print_coordinatesNIC();

	cout << WaveFct(atoms,a);

	cout << endl << endl;
*/
//	atoms.distanceNIC(0,1);
	cout << "End of execution!";
	getchar();
	return 0;
}
