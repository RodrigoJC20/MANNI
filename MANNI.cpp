// MANNI.cpp : Defines the entry point for the application.
//

#include "MANNI.h"

int main()
{
	std::string filename = "House.csv";
	std::string file_path = "../../../" + filename;

	arma::mat input;

	input.load(file_path);
	input.print("Matrix");

	/*
	double bias = 0;
	std::vector<double> w = { 0 };
	double alpha = 1e-2;

	LinearRegression::Model LinearModel;

	LinearModel.init_model_parameters(weights, w, bias, alpha);

	std::pair<double, double> final_w_b = LinearModel.gradient_descent(x, y, 10000);

	std::cout << "Final Model parameters w: " << final_w_b.first << " and b: " << final_w_b.second << "\n";

	std::vector<double> x_predictions = { 1, 1.5, 2, 2.5 };

	std::vector<double> results = LinearModel.compute_output(x_predictions);

	std::cout << "\nModel Predictions" << std::endl;

	for (size_t i = 0; i<results.size(); i++) {
		std::cout << "sqft: " << x_predictions[i] << " price: " << results[i] << "\n";
	}*/

	return 0;
}
