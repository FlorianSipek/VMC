VectorXd Jastrow::smooth_cutoff(double a, double rc1, double rc2){
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
	A(2,4) = 2;



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
	A(5,4) = 2;


	b[0] = (1-a/rc1)/(1-a/rc2);
	b[1] = (a/pow(rc1,2))/(1-a/rc2);
	b[2] = -2*(a/pow(rc1,3))/(1-a/rc2);
	b[3] = 1;

	VectorXd out = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
	cout << "output vector = " << out << endl;

	return out;
}
