#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

typedef std::chrono::steady_clock clk;

struct Individual {
    std::vector<bool> chromosome;
    double fitness = -1.0; // Cost function value
    double total_cost = -1.0; // price of bitstring

    bool operator<(const Individual& other) const {
        return fitness < other.fitness;
    }
};

class GeneticAlgorithm {
public:
    GeneticAlgorithm(
        const std::vector<double>& prices,
        const std::vector<int>& asset_bit_lengths,
        double budget,
        int population_size,
        double mutation_rate,
        double crossover_rate,
        int elitism_count,
        int tournament_size
    ) : n_assets(prices.size()),
        asset_bit_lengths(asset_bit_lengths),
        prices(prices),
        budget(budget),
        population_size(population_size),
        mutation_rate(mutation_rate),
        crossover_rate(crossover_rate),
        elitism_count(elitism_count),
        tournament_size(tournament_size),
        total_bits(0),
        rng(std::random_device{}())
        // rng(42)
    {
        start_ga = clk::now();
        if (prices.empty() || prices.size() != asset_bit_lengths.size()) {
            throw std::invalid_argument("Prices and asset_bit_lengths vectors must be non-empty and of the same size.");
        }
        if (elitism_count >= population_size) {
            throw std::invalid_argument("Elitism count must be less than population size.");
        }

        total_bits = std::accumulate(asset_bit_lengths.begin(), asset_bit_lengths.end(), 0);

        initialize_population();
    }

    void run(int generations, bool verbose = true) {
        for (int i = 0; i < generations; ++i) {
            evaluate_population();
            sort_population();
            evolve_new_generation();
            if ((i % 10 == 0 || i == generations - 1) && verbose) {
                std::cout << "Generation " << i << ": Best Fitness = " << std::scientific << population[0].fitness
                          << ", Cost = " << std::fixed << population[0].total_cost << std::endl;
            }
        }
        evaluate_population();
        sort_population();
    }

    Individual get_best_individual() const {
        if (population.empty() || population[0].fitness < 0) {
            throw std::runtime_error("Population has not been evaluated or is empty.");
        }
        // std::cout << "Price of 1st population: " << population[0].total_cost << "\n";
        // for (bool bit : population[0].chromosome) {
        //     std::cout << bit;
        // }
        // std::cout << "\n";
        // std::vector<bool> chromo_l = population[0].chromosome;
        // int cou = 1, i = 0;
        // while (true){
        //     bool equal = true;
        //     for (int j = 0; j < chromo_l.size(); ++j) {
        //         if (population[i].chromosome[j] != chromo_l[j]) {
        //             equal = false;
        //             break;
        //         }
        //     }
        //     if (!equal) {
        //         ++cou;
        //         chromo_l = population[i].chromosome;
        //     }
        //     if (cou >= 24)
        //         break;
        //     else
        //         ++i;
        // }
        // std::cout << "Price of 24th population: " << population[i].total_cost << "\n";
        // for (bool bit : population[i].chromosome) {
        //     std::cout << bit;
        // }
        // std::cout << "\n";
        // std::cout << "i: " << i << "\n";
        return population[0];
    }

    std::vector<Individual> get_top_n_individuals(int n, bool report=false) const {
        std::vector<Individual> top_individuals;
        n = std::min(n, static_cast<int>(population.size()));
        top_individuals.reserve(n);
        std::vector<bool> chromo_l = population[0].chromosome;
        top_individuals.push_back(population[0]);
        int cou = 1, i = 1;
        while (cou < n){
            bool equal = true;
            for (int j = 0; j < chromo_l.size(); ++j) {
                if (population[i].chromosome[j] != chromo_l[j]) {
                    equal = false;
                    break;
                }
            }
            if (!equal) {
                ++cou;
                chromo_l = population[i].chromosome;
                top_individuals.push_back(population[i]);
            }
            if (++i >= population.size())
                break;
        }
        if (report) {
            std::cout << "Price of " << n << "th population: " << top_individuals.back().total_cost << "(" << std::sqrt(top_individuals.back().fitness)/budget << ")\n";
            for (bool bit : top_individuals.back().chromosome) {
                std::cout << bit;
            }
            std::cout << "\n";
            finish_ga = clk::now();
            std::cout << "GA Execution Time:\n";
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(finish_ga - start_ga).count() << "ms\n";
            std::cout << std::chrono::duration_cast<std::chrono::microseconds>(finish_ga - start_ga).count() << "us\n";
        }
        return top_individuals;
    }

    std::vector<Individual> get_top_n_brute_force_individuals(int n, bool report = false) const {
        std::vector<Individual> tmp_population = population, top_individuals;
        population.clear();
        clk::time_point st_bf = clk::now();
        population.reserve(1 << total_bits);
        for (int i = 0; i < (1 << total_bits); ++i) {
            Individual individual;
            individual.chromosome.resize(total_bits);
            for (int j = 0; j < total_bits; ++j) {
                individual.chromosome[j] = (i >> j) & 1;
            }
            population.push_back(individual);
        }
        evaluate_population();
        sort_population();
        n = std::min(n, static_cast<int>(population.size()));
        top_individuals.reserve(n);
        for (int i = 0; i < n; ++i) {
            top_individuals.push_back(population[i]);
        }
        clk::time_point en_bf = clk::now();
        if (report) {
            std::cout << "Brute Force Execution Time:\n";
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(en_bf - st_bf).count() << "ms\n";
            std::cout << std::chrono::duration_cast<std::chrono::microseconds>(en_bf - st_bf).count() << "us\n";
            std::cout << "Price of " << n << "th population: " << top_individuals.back().total_cost << "(" << std::sqrt(top_individuals.back().fitness)/budget << ")\n";
            for (bool bit : top_individuals.back().chromosome) {
                std::cout << bit;
            }
            std::cout << "\n";
        }
        population = tmp_population;
        return top_individuals;
    }

private:
    int n_assets;
    std::vector<double> prices;
    std::vector<int> asset_bit_lengths;
    int total_bits;
    double budget;
    int population_size;
    double mutation_rate;
    double crossover_rate;
    int elitism_count;
    int tournament_size;
    mutable clk::time_point start_ga, finish_ga;

    mutable std::vector<Individual> population;
    std::mt19937 rng;

    void initialize_population() {
        population.resize(population_size);
        std::uniform_int_distribution<int> dist(0, 1);
        for (int i = 0; i < population_size; ++i) {
            population[i].chromosome.resize(total_bits);
            for (int j = 0; j < total_bits; ++j) {
                population[i].chromosome[j] = dist(rng);
            }
        }
    }

    void evaluate_population() const {
        for (auto& individual : population) {
            calculate_fitness(individual);
        }
    }

    void calculate_fitness(Individual& individual) const {
        double current_total_cost = 0.0;
        int curr_bit_idx = 0;
        for (int i = 0; i < n_assets; ++i) {
            int quantity = 0;
            for (int b = 0; b < asset_bit_lengths[i]; ++b) {
                quantity = (quantity << 1) | individual.chromosome[curr_bit_idx + b];
            }
            current_total_cost += quantity * prices[i];
            curr_bit_idx += asset_bit_lengths[i];
        }
        individual.total_cost = current_total_cost;

        double diff = current_total_cost - budget;
        individual.fitness = diff * diff;
    }

    void sort_population() const {
        std::sort(population.begin(), population.end());
    }
    
    void evolve_new_generation() {
        std::vector<Individual> new_population;
        new_population.reserve(population_size);

        for (int i = 0; i < elitism_count; ++i) {
            new_population.push_back(population[i]);
        }

        while (new_population.size() < population_size) {
            Individual parent1 = tournament_selection();
            Individual parent2 = tournament_selection();
            Individual child1 = parent1;
            Individual child2 = parent2;

            std::uniform_real_distribution<double> cross_dist(0.0, 1.0);
            if (cross_dist(rng) < crossover_rate) {
                single_point_crossover(parent1, parent2, child1, child2);
            }

            mutate(child1);
            mutate(child2);

            new_population.push_back(child1);
            if (new_population.size() < population_size) {
                new_population.push_back(child2);
            }
        }
        population = new_population;
    }
    
    Individual tournament_selection() {
        std::uniform_int_distribution<int> dist(0, population_size - 1);
        Individual best_in_tournament = population[dist(rng)];
        
        for (int i = 1; i < tournament_size; ++i) {
            const auto& contender = population[dist(rng)];
            if (contender.fitness < best_in_tournament.fitness) {
                best_in_tournament = contender;
            }
        }
        return best_in_tournament;
    }
    
    void single_point_crossover(const Individual& p1, const Individual& p2, Individual& c1, Individual& c2) {
        if (total_bits < 2) return;
        std::uniform_int_distribution<int> dist(1, total_bits - 1);
        int crossover_point = dist(rng);
        
        std::copy(p1.chromosome.begin(), p1.chromosome.begin() + crossover_point, c1.chromosome.begin());
        std::copy(p2.chromosome.begin() + crossover_point, p2.chromosome.end(), c1.chromosome.begin() + crossover_point);
        
        std::copy(p2.chromosome.begin(), p2.chromosome.begin() + crossover_point, c2.chromosome.begin());
        std::copy(p1.chromosome.begin() + crossover_point, p1.chromosome.end(), c2.chromosome.begin() + crossover_point);
    }

    void mutate(Individual& individual) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (int i = 0; i < total_bits; ++i) {
            if (dist(rng) < mutation_rate) {
                individual.chromosome[i] = !individual.chromosome[i];
            }
        }
    }
};

std::vector<int> decode_chromosome_to_quantities(const std::vector<bool>& chromosome, const std::vector<int>& asset_bit_lengths, int total_bits) {
    std::vector<int> quantities(asset_bit_lengths.size(), 0);
    int curr_bit_idx = 0;
    for (size_t i = 0; i < asset_bit_lengths.size(); ++i) {
        for (int b = 0; b < asset_bit_lengths[i]; ++b) {
            if (curr_bit_idx + b < chromosome.size()) {
                quantities[i] = (quantities[i] << 1) | chromosome[curr_bit_idx + b];
            }
        }
        curr_bit_idx += asset_bit_lengths[i];
    }
    return quantities;
}


int main() {
    clk::time_point st = clk::now();

    // int num_assets = 50;
    // double budget = 10000.50; 

    // int num_assets = 6;
    // double budget = 381.0143111171548; 

    int num_assets = 9;
    double budget = 380.17033271846947; 
    
    std::vector<int> asset_bit_lengths;
    for(int i = 0; i < num_assets; ++i) {
        asset_bit_lengths.push_back( (i % 3 == 0) ? 3 : 2 );
    }
    
    // std::vector<double> prices;
    // std::mt19937 price_rng(42);
    // std::uniform_real_distribution<double> price_dist(20.15, 250.85);
    // for(int i = 0; i < num_assets; ++i) {
        //     prices.push_back(price_dist(price_rng));
        // }
    // std::vector<double> prices = {95.37000275, 169.58999634, 132.3500061, 109.72000122, 103.02999878, 178.19000244};
    std::vector<double> prices = {131.30000305, 165.86000061, 183.25999451, 168.47000122, 152.94000244, 95.37000275, 110.30999756, 176.30000305, 103.02999878};

    int population_size = 1000;
    int generations = 100;
    double mutation_rate = 1.5 / double(std::accumulate(asset_bit_lengths.begin(), asset_bit_lengths.end(), 0));
    // double mutation_rate = 0.015;
    double crossover_rate = 0.85;
    int elitism_count = 2;
    int tournament_size = 5;

    // start_ga = clk::now();
    GeneticAlgorithm ga(prices, asset_bit_lengths, budget, population_size, mutation_rate, crossover_rate, elitism_count, tournament_size);
    ga.run(generations);
    std::vector<Individual> top_individuals = ga.get_top_n_individuals(24, true);
    clk::time_point en = clk::now();
    
    Individual best_solution = ga.get_best_individual();
    std::cout << "Actual distinct solutions found: " << top_individuals.size() << "\n";
    int total_bits = std::accumulate(asset_bit_lengths.begin(), asset_bit_lengths.end(), 0);
    std::vector<int> quantities = decode_chromosome_to_quantities(best_solution.chromosome, asset_bit_lengths, total_bits);

    // std::cout << "\n\n----------- Optimal Solution Found -----------\n" << std::endl;
    // std::cout << std::fixed << std::setprecision(8); 
    // std::cout << "Best Fitness (Objective Value): " << std::scientific << best_solution.fitness << std::fixed << std::endl;
    // std::cout << "Target Budget: " << budget << std::endl;
    // std::cout << "Achieved Cost: " << best_solution.total_cost << std::endl;
    // std::cout << "Difference:    " << std::abs(best_solution.total_cost - budget) << std::endl;
    
    // std::cout << "\nAsset Quantities to Buy:" << std::endl;
    // std::cout << std::setprecision(2);
    // for(int i = 0; i < num_assets; ++i) {
    //     if (quantities[i] > 0) {
    //             std::cout << "  - Asset " << i << " (Price: " << prices[i] << "): Buy " << quantities[i] << " units." << std::endl;
    //     }
    // }


    // std::cout << "Genetic Algorithm Execution Time:\n";
    // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(en - st).count() << "ms\n";
    // std::cout << std::chrono::duration_cast<std::chrono::microseconds>(en - st).count() << "us\n";


    ga.get_top_n_brute_force_individuals(24, true);



    return 0;
}

PYBIND11_MODULE(ga_solver, m) {
    m.doc() = "pybind11 plugin for a Genetic Algorithm Portfolio Optimizer";

    py::class_<Individual>(m, "Individual")
        .def(py::init<>())
        .def_readwrite("chromosome", &Individual::chromosome)
        .def_readwrite("fitness", &Individual::fitness)
        .def_readwrite("total_cost", &Individual::total_cost)
        .def("__repr__", [](const Individual &i) {
            return "<ga_solver.Individual fitness=" + std::to_string(i.fitness) +
                   " cost=" + std::to_string(i.total_cost) + ">";
        });

    py::class_<GeneticAlgorithm>(m, "GeneticAlgorithm")
        .def(py::init<const std::vector<double>&, const std::vector<int>&, double, int, double, double, int, int>(),
             py::arg("prices"),
             py::arg("asset_bit_lengths"),
             py::arg("budget"),
             py::arg("population_size"),
             py::arg("mutation_rate"),
             py::arg("crossover_rate"),
             py::arg("elitism_count"),
             py::arg("tournament_size"))
        .def("run", &GeneticAlgorithm::run,
             py::arg("generations"), py::arg("verbose") = true,
             "Run the genetic algorithm for a number of generations.")
        .def("get_best_individual", &GeneticAlgorithm::get_best_individual,
             "Get the best individual from the population.")
        .def("get_top_n_individuals", &GeneticAlgorithm::get_top_n_individuals,
             py::arg("n"), py::arg("report") = false,
             "Get the top N distinct individuals from the GA population.")
        .def("get_top_n_brute_force_individuals", &GeneticAlgorithm::get_top_n_brute_force_individuals,
             py::arg("n"), py::arg("report") = false,
             "Get the top N individuals using a brute-force approach for ground truth.");

    m.def("decode_chromosome_to_quantities", &decode_chromosome_to_quantities,
          py::arg("chromosome"),
          py::arg("asset_bit_lengths"),
          py::arg("total_bits"),
          "Decode a chromosome bitstring into integer quantities for each asset.");
}