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

	std::vector<long double> x, y;
	double sqft, price;

	std::cout << "Real prices:" << std::endl;

	while (in.read_row(sqft, price)) {
		std::cout << "sqft: " << sqft << " price: " << price << "\n";
		x.emplace_back(sqft);
		y.emplace_back(price);
	}

	long double bias = 100;
	std::vector<long double> w = { 200 };

	LinearRegression::Model LinearModel;

	LinearModel.set_model_parameters(w, bias);

	std::vector<long double> results = LinearModel.compute_output(x);

	std::cout << "\nModel Predictions" << std::endl;

	for (int i = 0; i<results.size(); i++) {
		std::cout << "sqft: " << x[i] << " price: " << results[i] << "\n";
	}

	return 0;
}
