#include "Plot.h"

#include <fstream>

void Plot::ExportAverageRewardsOverEpisodes(const std::vector<std::vector<float>>& data, const std::string& filename) {
    std::ofstream outFile("../" + filename + ".csv");

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            outFile << row[i];
            if (i < row.size() - 1) outFile << ",";
        }
        outFile << "\n";
    }

    outFile.close();
}
