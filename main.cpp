#include <iostream>

#include "Agent/QLAgent.h"
#include "Agent/RandomAgent.h"
#include "DecayScheduler/LinearDecayScheduler.h"
#include "Environment/Environment.h"
#include "Utils/Plot.h"

#include <numeric>

int main() {
    const unsigned seed = 123;
    const unsigned numProc = 4;
    const unsigned numThread = 2;
    const unsigned numTask = 20;
    const unsigned maxTaskDuration = 10;

    const float epsilonMin = 0.1;
    const float epsilonMax = 1.0;
    const float epsilonDecayRate = 0.001;

    const float alpha = 0.5;
    const float gamma = 0.9;

    const unsigned numRun = 4;
    const unsigned numEpisode = 100000;

    // Data for plotting
    std::vector<std::vector<int>> qlRewards;
    float qlUtilScore = 0.0;

    std::vector<std::vector<int>> randRewards;
    float randSteps = 0.0;

    // Running the train and rollout
    const auto env = std::make_shared<Environment>(numProc, numThread, maxTaskDuration, numTask, seed);
    const auto ds = std::make_shared<LinearDecayScheduler>(epsilonMin, epsilonMax, epsilonDecayRate);

    // Run Random
    std::unique_ptr<RandomAgent> randAgent;
    for (unsigned i = 0; i < numRun; i++) {
        std::vector<int> runRewards;
        unsigned runSteps;
        randAgent = std::make_unique<RandomAgent>(env, seed);

        std::tie(runRewards, runSteps) = randAgent->rollout(numEpisode);
        randRewards.push_back(runRewards);
        randSteps += static_cast<float>(runSteps);
    }

    randSteps = std::floor(randSteps / numRun);

    // Run Q Learning
    std::unique_ptr<QLAgent> dlAgent;
    for (unsigned i = 0; i < numRun; i++) {
        std::vector<int> runRewards;
        float uScore;
        dlAgent = std::make_unique<QLAgent>(env, alpha, gamma, ds, seed);

        std::tie(runRewards, uScore) = dlAgent->train(numEpisode);
        qlRewards.push_back(runRewards);
        qlUtilScore += uScore;
    }

    const unsigned qlSteps = dlAgent->rollout();
    qlUtilScore = qlUtilScore / numRun;

    // Result
    std::cout << "Random Policy: " << randSteps << " steps" << std::endl;
    std::cout << "Q Learning Policy: " << qlSteps << " steps | " << qlUtilScore << " Avg Utilization" << std::endl;

    Plot::ExportAverageRewardsOverEpisodes(qlRewards, "ql");
    Plot::ExportAverageRewardsOverEpisodes(randRewards, "rand");
    return 0;
}
