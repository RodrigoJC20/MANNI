// MANNI.cpp : Defines the entry point for the application.
//

#include "MANNI.h"

int main()
{
	std::string filename = "MultipleLinearRegression.csv";
	std::string file_path = "../../../examples/" + filename;

	arma::mat input;

	input.load(file_path);
	input.print("Input Dataset: ");

	arma::vec y_train = input.col(input.n_cols - 1);
	arma::mat x_train = input.head_cols(input.n_cols - 1);

	y_train.print("y_train: ");
	x_train.print("x_train: ");

	//arma::vec w_init = arma::randu(x_train.n_cols, arma::distr_param(0, 200));
	//double b_init = arma::randu(arma::distr_param(0, 200));
	double learning_rate = 1e-2;

	double b_init = 785.1811367994083;
	arma::vec w_init = { 0.39133535, 18.75376741, -53.36032453, -26.42131618 };

	w_init.print("w_init: ");
	std::cout << std::setprecision(12);
	std::cout << "b_init: " << b_init << std::endl;
	std::cout << "alpha: " << learning_rate << std::endl;

	LinearRegression::Model LinearModel (w_init, b_init, learning_rate);

	double cost = LinearModel.compte_cost(x_train, y_train);

	std::cout << "Cost: " << cost << std::endl;

	arma::rowvec x_vec = x_train.row(0);

	double prediction = LinearModel.predict_single_output(x_vec);

	std::cout << prediction << std::endl;

	return 0;
}