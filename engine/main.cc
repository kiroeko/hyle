#include "application.h"
#include "model_catalog.h"

#include <iostream>

int main(int argc, char **argv) {
  return hyle::engine::run_application(hyle::engine::model_providers(), argc, argv, std::cin, std::cout, std::cerr);
}
