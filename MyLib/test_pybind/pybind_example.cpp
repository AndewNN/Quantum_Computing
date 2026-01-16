#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void say_hello() {
    std::cout << "Hello from C++!" << std::endl;
}

double multiply(double a, double b) {
    return a * b;
}

PYBIND11_MODULE(pybind_example, m) {
    m.doc() = "A basic pybind11 example";

    m.def("say_hello", &say_hello, "A function that prints 'Hello from C++'");

    m.def("multiply", &multiply, "A function that multiplies two numbers", py::arg("a"), py::arg("b"));
}