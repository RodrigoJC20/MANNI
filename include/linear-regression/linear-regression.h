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

		Model(const arma::vec& w, const double& b, const double& alpha)
			: weights(w), bias(b), alpha(alpha) {}


		/*
		* @brief Compute the prediction of the for a single vector
		*
		* @param x :- Input vector
		*
		* @return Results of the prediction
		*/
		double single_prediction(const arma::vec& x) const { 
			return arma::dot(x, weights) + bias;
		}

		double single_prediction(const arma::rowvec& x) const {
			return arma::dot(x, weights) + bias;
		}

		/*
		* @brief Perform feature scaling (0,1) for every feature.
		*/
		void performFeatureScaling(arma::mat& x) {
			for (size_t i = 0; i < x.n_cols; i++) {
				//get max
				double max = std::numeric_limits<double>::min();
				double min = std::numeric_limits<double>::max();
				for (size_t j = 0; j < x.n_rows; j++) {
					max = std::max(x.at(j, i), max);
					min = std::min(x.at(j, i), min);
				}
				//aply changes
				//x.col(i) /= max;
				for (size_t j = 0; j < x.n_rows; j++) {
					//x.at(i, j) = (x.at(i, j) - min) / (max - min);
					x.at(j, i) = x.at(j, i) / max;
				}
			}
		}

		double compte_cost(const arma::mat& x, const arma::vec& y) {
			int m = x.n_rows; 
			double cost = 0;

			for (int i = 0; i < m; i++) {
				double prediction = arma::dot(x.row(i), weights) + bias;
				double error = prediction - y[i];
				cost += error * error;
			}

			return cost / (2 * m);
		}

		std::pair<arma::vec, double> compute_gradient(const arma::mat& x, const arma::vec& y) const {
			int m = x.n_rows; 
			int n = x.n_cols;
			arma::vec dcost_w = arma::zeros(n);
			double dcost_b = 0;	
			
			for (int i = 0; i < m; i++) {
				double error = (arma::dot(x.row(i), weights) + bias) - y[i];
				for (int j = 0; j < n; j++) {
					dcost_w[j] = dcost_w[j] + error * x(i, j);
				}
				dcost_b = dcost_b + error;
			}

			dcost_w /= m;
			dcost_b /= m;

			return { dcost_w, dcost_b };
		}

		/*
		* @brief Performs gradient descent to find w, b.
		*
		* @param x -: Data, m examples as a vector of doubles
		* @param y -: Target values as a vector of doubles
		* @param iterations -: Number of iterations to run gradient descent
		*
		* @return Pair of updates values for w and b after running gradient descent
		*/
		std::pair<arma::vec, double> gradient_descent(const arma::mat& x, const arma::vec& y, const int& iterations, const bool suppress_output = false) {
			if (!suppress_output) std::cout << "\n Computing Gradient Descent..." << std::endl;
			int n = x.n_cols;

			

			for (size_t it = 0; it < iterations; it++) {
				
				
				std::pair<arma::vec, double> gradient = compute_gradient(x, y);

				gradient.first *= alpha;

				for (int i=0; i<n; i++) weights[i] = weights[i] - gradient.first[i];
				
				bias = bias - gradient.second * alpha;

				if (it % 100 == 0 && !suppress_output) {
					double cost = compute_cost(x, y);
					std::cout << std::setw(10) << "Iteration" << std::setw(8) << it << ":";
					std::cout << std::setw(6) << "Cost" << std::setw(14) << cost << "\n";
				}
			}

			if (!suppress_output) std::cout << "\n";

			return { weights, bias };
		}

		/*
		* @brief Set the parameters of the linear regression model
		* 
		* @param w -: The model parameters (weights) as a vector of doubles
		* @param b -: The bias term (intercept) of the model
		* @para alpha -: Learning rate alpha
		*/
		void init_model_parameters(const arma::vec& w, const double& b, const double& alpha) {
			this->weights = w;
			this->bias = b;
			this->alpha = alpha;
		}

		void print_model_params() {
			std::cout << "Weights: [ ";
			for (double w : weights) {
				std::cout << w << " ";
			}
			std::cout << "]" << std::endl;
			std::cout << "Bias: " << bias << std::endl;
		}

		void print_model_params(std::string text) {
			std::cout << text << std::endl;
			std::cout << "Weights: [ ";
			for (double w : weights) {
				std::cout << w << " ";
			}
			std::cout << "]" << std::endl;
			std::cout << "Bias: " << bias << std::endl;
		}

	private:
		arma::vec weights;
		double bias;
		double alpha;
	};
}

#endif // !LINEAR_REGRESSION_H
