# include<bits/stdc++.h>
using namespace std;

struct Individual{
    int index;
    double fitness;
    double total_cost;

    bool operator<(const Individual &other) const {
        if (fitness != other.fitness)
            return fitness < other.fitness;
        if (total_cost != other.total_cost)
            return total_cost < other.total_cost;
        return index < other.index;
    }
};

typedef chrono::steady_clock clk;

clk::time_point start_bf = clk::now();
double budget = 351.789;
mt19937 rng(919);
uniform_real_distribution<double> dist(100.0, 200.0);
int assets = 11, qubits_per_asset = 2;
vector<Individual> all_individuals(1<<(assets * qubits_per_asset));

int main(){
    vector<double> prices(assets);
    for(int i = 0; i < assets; ++i)
        prices[i] = dist(rng);
    for(int i = 0; i < (1<<(assets * qubits_per_asset)); ++i){
        double total_cost = 0.0;
        for(int j = 0; j < assets; ++j){
            int quantity = 0;
            for(int b = 0; b < qubits_per_asset; ++b){
                if( (i & (1 << (j * qubits_per_asset + b))) != 0 )
                    quantity |= (1 << b);
            }
            total_cost += quantity * prices[j];
        }
        double diff = total_cost - budget;
        double fitness = diff * diff;
        all_individuals[i] = {i, fitness, total_cost};
    }
    sort(all_individuals.begin(), all_individuals.end());
    clk::time_point finish_bf = clk::now();
    // cout << "Top 10 solutions found by brute-force:\n";
    // for(int i = 0; i < 10; ++i){
    //     const auto &ind = all_individuals[i];
    //     cout << "Solution " << i+1 << ": Index = " << ind.index 
    //          << ", Fitness = " << scientific << ind.fitness << fixed
    //          << ", Total Cost = " << ind.total_cost << "\n";
    //     cout << "  Chromosome bits: ";
    //     for(int b = assets * qubits_per_asset - 1; b >= 0; --b){
    //         bool bit = (ind.index & (1 << b)) != 0;
    //         cout << bit;
    //     }
    //     cout << "\n";
    // }
    cout << "Brute-force Execution Time:\n";
    cout << chrono::duration_cast<chrono::milliseconds>(finish_bf - start_bf).count() << "ms\n";
    cout << chrono::duration_cast<chrono::microseconds>(finish_bf - start_bf).count() << "us\n";
    return 0;
}