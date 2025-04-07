#include "Plot.h"

#include <matplot/matplot.h>
#include <torch/torch.h>

void Plot::ExportAverageRewardsOverEpisodes(const std::vector<std::vector<int>>& data, float baseline) {
    std::ofstream outFile("../data.csv");

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) outFile << ",";
        }
        outFile << "\n";
    }

    outFile.close();
}
