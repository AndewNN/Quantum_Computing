import pybind_example

print("Calling the C++ function to print a message...")
pybind_example.say_hello()

a = 12.5
b = 4.0
result = pybind_example.multiply(a=a, b=b)

print(f"\nCalling the C++ function to multiply {a} and {b}...")
print(f"Result returned from C++: {result}")
print(f"Type of the result is: {type(result)}")
