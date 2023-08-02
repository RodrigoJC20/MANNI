// MANNI.cpp : Defines the entry point for the application.
//

#include "MANNI.h"

int main()
{
	std::string filename = "House.csv";
	std::string path = "../../../" + filename;

	io::CSVReader<2> in(path);
	in.read_header(io::ignore_extra_column, "sqmt", "price");
	int sqmt, price;
	while (in.read_row(sqmt, price)) {
		std::cout << "sqmt: " << sqmt << " price: " << price << "\n";
	}

	return 0;
}
