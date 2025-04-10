#include "Agent/QLAgent.h"
#include "Agent/RandomAgent.h"
#include "DecayScheduler/LinearDecayScheduler.h"
#include "Environment/Environment.h"
#include "Utils/Plot.h"

#include <numeric>

int main() {
    const unsigned seed = 123;
    const unsigned numProc = 2;
    const unsigned numThread = 3;
    const unsigned maxTaskDuration = 50;

    const float epsilonMin = 0.1;
    const float epsilonMax = 1.0;
    const float epsilonDecayRate = 0.001;

    const float alpha = 0.5;
    const float gamma = 0.9;

    const unsigned numRun = 4;
    const unsigned numEpisode = 30000;

    // Data for plotting
    std::vector<std::vector<int>> qlRewards;
    std::vector<std::vector<int>> randRewards;
    double randSteps = 0.0;

    // Running the train and rollout
    const auto env = std::make_shared<Environment>(numProc, numThread, maxTaskDuration, seed);
    const auto ds = std::make_shared<LinearDecayScheduler>(epsilonMin, epsilonMax, epsilonDecayRate);

    // Run Random
    std::unique_ptr<RandomAgent> randAgent;
    for (unsigned i = 0; i < numRun; i++) {
        std::vector<int> runRewards;
        unsigned runSteps = 0;
        randAgent = std::make_unique<RandomAgent>(env, seed);

        std::tie(runRewards, runSteps) = randAgent->rollout(numEpisode);
        randRewards.push_back(runRewards);
        randSteps += runSteps;
    }

    randSteps = std::floor(randSteps);

    // Run Q Learning
    std::unique_ptr<QLAgent> dlAgent;
    for (unsigned i = 0; i < numRun; i++) {
        dlAgent = std::make_unique<QLAgent>(env, alpha, gamma, ds, seed);
        qlRewards.push_back(dlAgent->train(numEpisode));
    }

    const unsigned qlSteps = dlAgent->rollout();

    std::cout << "Random Policy: " << randSteps << " steps" << std::endl;
    std::cout << "Q Learning Policy: " << qlSteps << " steps" << std::endl;

    Plot::ExportAverageRewardsOverEpisodes(qlRewards, "ql");
    Plot::ExportAverageRewardsOverEpisodes(randRewards, "rand");
    return 0;
}
