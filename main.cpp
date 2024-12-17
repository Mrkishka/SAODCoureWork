#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cmath>
#include <omp.h>

const double DAMPING_FACTOR = 0.85;
const double EPSILON = 1e-6;
const int MAX_ITERATIONS = 100;

// Функция для расчета PageRank
std::unordered_map<int, double> calculatePageRank(
    const std::vector<std::vector<int>>& graph,
    const std::unordered_map<int, double>& personalizedVector = {},
    const std::unordered_map<int, std::unordered_map<int, double>>& linkWeights = {}) {

    int numPages = graph.size();
    std::unordered_map<int, double> rank;
    std::unordered_map<int, double> newRank;

    for (int i = 0; i < numPages; i++) {
        rank[i] = 1.0 / numPages;
    }

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        for (int i = 0; i < numPages; i++) {
            newRank[i] = (1.0 - DAMPING_FACTOR) / numPages;
        }

        double danglingRank = 0.0;
        #pragma omp parallel for reduction(+:danglingRank)
        for (int i = 0; i < numPages; i++) {
            if (graph[i].empty()) {
                danglingRank += rank[i];
            }
        }

        #pragma omp parallel for
        for (int j = 0; j < numPages; j++) {
            newRank[j] += DAMPING_FACTOR * danglingRank / numPages;
        }

        #pragma omp parallel for
        for (int i = 0; i < numPages; i++) {
            int outDegree = graph[i].size();
            if (outDegree == 0) continue;

            for (int j : graph[i]) {
                double weight = linkWeights.count(i) && linkWeights.at(i).count(j) ? linkWeights.at(i).at(j) : 1.0;
                newRank[j] += DAMPING_FACTOR * rank[i] * weight / outDegree;
            }
        }

        if (!personalizedVector.empty()) {
            for (const auto& [page, weight] : personalizedVector) {
                newRank[page] += weight * (1.0 - DAMPING_FACTOR);
            }
        }

        double diff = 0.0;
        #pragma omp parallel for reduction(+:diff)
        for (int i = 0; i < numPages; i++) {
            diff += std::fabs(newRank[i] - rank[i]);
            rank[i] = newRank[i];
        }

        if (diff < EPSILON) {
            std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    return rank;
}

void exportGraphToDOT(const std::vector<std::vector<int>>& graph, const std::string& filename) {
    std::ofstream file(filename);
    file << "digraph G {\n";
    for (size_t i = 0; i < graph.size(); i++) {
        for (int j : graph[i]) {
            file << "  " << i << " -> " << j << ";\n";
        }
    }
    file << "}\n";
    file.close();
}

int main() {
    std::vector<std::vector<int>> graph = {
        {1, 2},   // Страница 0 ссылается на 1 и 2
        {2},      // Страница 1 ссылается на 2
        {0},      // Страница 2 ссылается на 0
        {0, 2}    // Страница 3 ссылается на 0 и 2
    };

    std::unordered_map<int, std::unordered_map<int, double>> linkWeights = {
        {0, {{1, 1.5}, {2, 1.0}}},
        {3, {{0, 2.0}, {2, 0.5}}}
    };

    std::unordered_map<int, double> personalizedVector = {
        {0, 0.5},
        {2, 0.3}
    };

    auto rank = calculatePageRank(graph, personalizedVector, linkWeights);

    std::cout << "PageRank values:" << std::endl;
    for (const auto& [page, value] : rank) {
        std::cout << "Page " << page << ": " << value << std::endl;
    }

    exportGraphToDOT(graph, "graph.dot");
    std::cout << "Граф сохранён в 'graph.dot'." << std::endl;

    return 0;
}
