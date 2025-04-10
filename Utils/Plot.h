#ifndef PLOT_H
#define PLOT_H
#include <string>
#include <vector>

class Plot {
public:
    Plot() = delete;
    Plot(const Plot&) = delete;
    Plot& operator = (const Plot&) = delete;

    static void ExportAverageRewardsOverEpisodes(const std::vector<std::vector<int>>& data, const std::string& filename);
};

#endif //PLOT_H
