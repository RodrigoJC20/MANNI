﻿// MANNI.cpp : Defines the entry point for the application.
//

#include "MANNI.h"

void test_open_blas() {
	arma::mat A = arma::randu<arma::mat>(3, 3); // Random 3x3 matrix
	arma::vec b = arma::randu<arma::vec>(3);    // Random vector

	arma::vec x = arma::solve(A, b);            // Solve Ax = b

	std::cout << "Solution x:\n" << x << std::endl;
}

int main()
{
	/*std::string filename = "MultipleLinearRegression.csv";
	std::string file_path = "../../../examples/" + filename;

	arma::mat input;

	input.load(file_path);
	input.print("Input Dataset: ");

	arma::vec y_train = input.col(input.n_cols - 1);
	arma::mat x_train = input.head_cols(input.n_cols - 1);

	y_train.print("y_train: ");
	x_train.print("x_train: ");

	arma::vec w_init = arma::randu(x_train.n_cols, arma::distr_param(0, 200));
	double b_init = arma::randu(arma::distr_param(0, 200));
	double learning_rate = 1e-2;

	// double b_init = 785.1811367994083;
	// arma::vec w_init = { 0.39133535, 18.75376741, -53.36032453, -26.42131618 };

	w_init.print("w_init: ");
	std::cout << "b_init: " << b_init << std::endl;
	std::cout << "alpha: " << learning_rate << std::endl;

	LinearRegression::Model LinearModel (w_init, b_init, learning_rate);

	double cost = LinearModel.compte_cost(x_train, y_train);

	std::cout << "Cost: " << cost << std::endl;
	*/
	
	test_open_blas();

	return 0;
}
