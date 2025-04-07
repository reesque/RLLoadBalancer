#ifndef PLOT_H
#define PLOT_H
#include <vector>

class Plot {
public:
    Plot() = delete;
    Plot(const Plot&) = delete;
    Plot& operator = (const Plot&) = delete;

    static void ExportAverageRewardsOverEpisodes(const std::vector<std::vector<int>>& data, float baseline);
};

#endif //PLOT_H
