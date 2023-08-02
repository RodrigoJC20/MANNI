// MANNI.cpp : Defines the entry point for the application.
//

#include "MANNI.h"
#include "csv.h"


int main()
{
	std::cout << "Hello World with CMake" << std::endl;
	io::CSVReader<2> in("C:\\Users\\soder\\source\\repos\\RodrigoJC20\\MANNI\\House.csv");
	in.read_header(io::ignore_extra_column, "sqmt", "price");
	int sqmt, price;
	while (in.read_row(sqmt, price)) {
		std::cout << "sqmt: " << sqmt << " price: " << price << "\n";
	}

	std::cout << std::endl;
	return 0;
}
