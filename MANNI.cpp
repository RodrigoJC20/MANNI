// MANNI.cpp : Defines the entry point for the application.
//

#include "MANNI.h"

int main()
{
	std::string filename = "MultipleLinearRegression.csv";
	std::string file_path = "../../../examples/" + filename;

	arma::mat input;

	input.load(file_path);

	arma::vec y_train = input.col(input.n_cols - 1);
	arma::mat x_train = input.head_cols(input.n_cols - 1);

	std::vector<std::pair<double, double>> x_mu_sigma = LinearRegression::Model::z_score_normalization(x_train);

	for (int i = 0; i < x_train.n_cols; i++) {
		x_train.col(i) -= x_mu_sigma[i].first;
		x_train.col(i) /= x_mu_sigma[i].second;
	}
	
	//arma::vec w_init(x_train.n_cols, arma::fill::zeros);
	//double b_init = 0;

	double b_init = 0;
	arma::vec w_init = arma::zeros(x_train.n_cols);
	double learning_rate = 5.0e-7;
	int iterations = 1000;

	std::cout << std::setprecision(6);

	LinearRegression::Model LinearModel (w_init, b_init, learning_rate);

	LinearModel.print_model_params("Initial w and b: ");
	std::cout << "Learning rate: " << learning_rate << std::endl;
	std::cout << "Iterations: " << iterations << std::endl;
	
	std::pair<arma::vec, double> result = LinearModel.gradient_descent(x_train, y_train, iterations);

	LinearModel.print_model_params("Final w and b found by gradient descent: ");
	
	int i = 0;
	double accuracy = 0.0;
	double tolerance = 10.0;
	int num_accurate_predictions = 0;

	std::cout << "\nTest against x_train: " << std::endl;
	x_train.each_row([&](const arma::rowvec& row) {
		double result = LinearModel.single_prediction(row);
		double diff = std::abs(result - y_train[i]);
		double deviation = (diff / y_train[i]) * 100.0;

		if (deviation <= tolerance) {
			num_accurate_predictions++;
		}

		std::cout << std::setw(14) << "Prediction:" << std::setw(8) << result;
		std::cout << std::setw(16) << "Target value:" << std::setw(4) << y_train[i++];
		std::cout << std::setw(8) << "Diff:" << std::setw(8) << diff;
		std::cout << std::setw(13) << "Deviation:" << std::setw(3) << "% " << deviation << std::endl;
	});

	double accuracy_percentage = static_cast<double>(num_accurate_predictions) / x_train.n_rows * 100.0;
	std::cout << "Accurate predictions: " << num_accurate_predictions << std::endl;
	std::cout << "Model accuracy: % " << accuracy_percentage << std::endl;
	
	return 0;
}