# jaxlob2
Limit Order Book Simulations in JAX

## Features
- Fast JIT compilation
- Easy parallelization
- LOBSTER data handling
- `black` and `isort` code formatting
- Appropriate variable names and typing
- Docstrings
- Testing to assert correct behaviour

## Module Guide
- `job`: "jax order book" contains pure functions to manipulate the state of the order book
- `dataloader`: functions to extract, read and process LOBSTER data
- `optimal_execution`: gymnax envrionments for the optimal execution problem

## Acknowledgements

Jaxlob2 is inspired by and built on top of [jaxlob](https://github.com/KangOxford/jax-lob)
