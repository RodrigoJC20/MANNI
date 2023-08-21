#pragma once
#ifndef LINEAR_REGRESSION_H

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <armadillo>

namespace LinearRegression {
	class Model {
	public:
		/*
		* @brief Set the parameters of the linear regression model
		*
		* @param w -: The model parameters (weights) as a vector of doubles
		* @param b -: The bias term (intercept) of the model
		* @para alpha -: Learning rate alpha
		*/
		Model(const arma::vec& w, const double& b, const double& alpha) {
			this->weights = w;
			this->bias = b;
			this->alpha = alpha;
		}

		/*
		* @brief Compute the prediction of the for a single vector
		*
		* @param x :- Input vector
		*
		* @return Results of the prediction
		*/
		double predict_single_output(arma::vec x) { 
			return arma::dot(x, this->weights) + this->bias;
		}

		double compte_cost(arma::mat x, arma::vec y) {
			double cost = 0;
			int rowIndex = 0;
			x.each_row([&](const arma::rowvec& row) {
				double cost_per_row = arma::dot(row.as_col(), this->weights) + this->bias;
				cost_per_row = (cost_per_row - y[rowIndex]) * (cost_per_row - y[rowIndex]);
				cost += cost_per_row;
				rowIndex++;
			});

			return cost / (2 * y.n_cols);
		}

		/*
		* @brief Compute the cost (loss)
		* 
		* @param x -: Data, m examples as a vector of doubles
		* @param y -: Target values as a vector of doubles
		* 
		* @return The cost of the model as a double
		*/
		/*double compute_cost(const std::vector<double>& x, const std::vector<double>& y) {
			size_t m = y.size();

			double acum = 0;
			for (size_t i = 0; i < m; ++i) {
				double line = w[0] * x[i] + b;
				double diff = (line - y[i]) * (line - y[i]);
				acum += diff;
			}

			return acum / (2 * m);
		}*/

		/*
		* @brief Computes the gradient for linear regression
		* 
		* @param x -: Data, m examples as a vector of doubles
		* @param y -: Target values as a vector of doubles
		* 
		* @return A pair of the gradient of the partial derivates with respect to w and b as a pair of doubles
		*		  Pair<Gradient of the cost w.r.t w, Gradient of the cost w.r.t b> 
		*/
		/*std::pair<double, double> compute_gradient(const std::vector<double>& x, const std::vector<double>& y) {
			size_t m = x.size();
			double dcost_w = 0;
			double dcost_b = 0;

			for (size_t i = 0; i < m; ++i) {
				double f_wb = w[0] * x[i] + b;
				double dcost_w_pre = (f_wb - y[i]) * x[i];
				double dcost_b_pre = f_wb - y[i];

				dcost_w += dcost_w_pre;
				dcost_b += dcost_b_pre;
			}

			dcost_b /= m;
			dcost_w /= m;

			return { dcost_w, dcost_b };
		}*/

		/*
		* @brief Performs gradient descent to find w, b.
		* 
		* @param x -: Data, m examples as a vector of doubles
		* @param y -: Target values as a vector of doubles
		* @param iterations -: Number of iterations to run gradient descent
		* 
		* @return Pair of updates values for w and b after running gradient descent
		*/
		/*std::pair<double, double> gradient_descent(const std::vector<double>& x, const std::vector<double>& y, const int& iterations) {
			std::cout << std::left << std::setw(12) << "Iteration"
				<< std::setw(15) << "Cost"
				<< std::setw(15) << "dcost_w"
				<< std::setw(15) << "dcost_b"
				<< std::setw(15) << "w"
				<< std::setw(15) << "b"
				<< "\n";

			for (size_t it = 0; it < iterations; ++it) {
				std::pair<double, double> dcost = compute_gradient(x, y);

				w[0] = w[0] - alpha * dcost.first;
				b = b - alpha * dcost.second;

				if (it % 1000 == 0) {
					double cost = compute_cost(x, y);
					
					std::cout << std::left << std::setw(12) << it
						<< std::setw(15) << std::fixed << std::setprecision(6) << cost
						<< std::setw(15) << std::fixed << std::setprecision(6) << dcost.first
						<< std::setw(15) << std::fixed << std::setprecision(6) << dcost.second
						<< std::setw(15) << std::fixed << std::setprecision(6) << w[0]
						<< std::setw(15) << std::fixed << std::setprecision(6) << b
						<< "\n";
				}
			}

			return { w[0], b };
		}*/

		/*
		* @brief Set the parameters of the linear regression model
		* 
		* @param w -: The model parameters (weights) as a vector of doubles
		* @param b -: The bias term (intercept) of the model
		* @para alpha -: Learning rate alpha
		*/
		void init_model_parameters(const arma::vec& w, const double b, const double& alpha) {
			this->weights = w;
			this->bias = b;
			this->alpha = alpha;
		}

	private:
		arma::vec weights;
		double bias;
		double alpha;
	};
}

#endif // !LINEAR_REGRESSION_H
