#pragma once
#ifndef LINEAR_REGRESSION_H

#include <iostream>
#include <vector>
#include <string>

namespace LinearRegression {
	class Model {
	public:
		/*
		* @brief Compute the prediction of the linear model
		* 
		* @param x :- Input data, m examples
		* 
		* @return Results of y-hat predictions as a vector of doubles
		* f(x) = w*x + b
		*/
		std::vector<double> compute_output(const std::vector<double>& x) {
			size_t m = x.size();

			std::vector<double> f_predictions(m, 0);

			for (size_t i = 0; i < m; i++) {
				f_predictions[i] = w[0] * x[i] + b;
			}

			return f_predictions;
		}

		double loss(const std::vector<double>& x, const std::vector<double>& y) {
			size_t m = y.size();

			double acum = 0;
			for (size_t i = 0; i < m; ++i) {
				double line = w[0] * x[i] + b;
				double diff = (line - y[i]) * (line - y[i]);
				acum += diff;
			}

			return acum / (2 * m);
		}

		/*\
		* @brief Set the parameters of the linear regression model
		* 
		* @param w -: The model parameters (weights) as a vector of doubles
		* @param b -: The bias term (intercept) of the model
		*/
		void set_model_parameters(const std::vector<double>& model_params, const double& b) {
			this->w = model_params;
			this->b = b;
		}

	private:
		std::vector<double> w;
		double b;
	};
}

#endif // !LINEAR_REGRESSION_H
