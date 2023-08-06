// MANNI.cpp : Defines the entry point for the application.
//

#include "MANNI.h"
#include "linear-regression/linear-regression.h"

int main()
{
	std::string filename = "House.csv";
	std::string path = "../../../" + filename;

	io::CSVReader<2> in(path);
	in.read_header(io::ignore_extra_column, "sqft", "price");

	std::vector<double> x, y;
	double sqft, price;

	std::cout << "Actual:" << std::endl;

	while (in.read_row(sqft, price)) {
		std::cout << "sqft: " << sqft << " price: " << price << "\n";
		x.emplace_back(sqft);
		y.emplace_back(price);
	}
	std::cout << "\n";

	double bias = 0;
	std::vector<double> w = { 0 };
	double alpha = 1e-2;

	LinearRegression::Model LinearModel;

	LinearModel.init_model_parameters(w, bias, alpha);

	std::pair<double, double> final_w_b = LinearModel.gradient_descent(x, y, 10000);

	std::cout << "Final Model parameters w: " << final_w_b.first << " and b: " << final_w_b.second << "\n";

	std::vector<double> x_predictions = { 1, 1.5, 2, 2.5 };
	
	std::vector<double> results = LinearModel.compute_output(x_predictions);

	std::cout << "\nModel Predictions" << std::endl;

	for (size_t i = 0; i<results.size(); i++) {
		std::cout << "sqft: " << x_predictions[i] << " price: " << results[i] << "\n";
	}
	
	return 0;
}
