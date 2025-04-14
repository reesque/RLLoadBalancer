#include <iostream>

#include "Agent/QLAgent.h"
#include "Agent/RandomAgent.h"
#include "Agent/DQNAgent.h"
#include "DecayScheduler/LinearDecayScheduler.h"
#include "DecayScheduler/ExponentialDecayScheduler.h"
#include "Environment/Environment.h"
#include "Utils/Plot.h"

#include <numeric>

int main() {
    const unsigned seed = 123;
    const unsigned numProc = 4;
    const unsigned numThread = 2;
    const unsigned numTask = 10;
    const unsigned maxTaskDuration = 20;

    const float epsilonMin = 0.1;
    const float epsilonMax = 1.0;
    const float epsilonDecayRate = 0.0001;

    const float alpha = 0.5;
    const float gamma = 0.9;
    const float lambda = 0.1;

    const unsigned numRun = 1;
    const unsigned numEpisode = 10000;

    // Data for plotting
    std::vector<std::vector<float>> qlRewards;

    //std::vector<std::vector<int>> randRewards;
    //float randSteps = 0.0;

    // Running the train and rollout
    const auto env = std::make_shared<Environment>(numProc, numThread, maxTaskDuration, numTask, lambda, seed);
    const auto ds = std::make_shared<LinearDecayScheduler>(epsilonMin, epsilonMax, epsilonDecayRate);

    /*
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

    randSteps = std::floor(randSteps / numRun);*/

    // Run Q Learning
    std::unique_ptr<QLAgent> dlAgent;
    for (unsigned i = 0; i < numRun; i++) {
        dlAgent = std::make_unique<QLAgent>(env, alpha, gamma, ds, seed);
        qlRewards.push_back(dlAgent->train(numEpisode));
    }

    auto [qlSteps, qlUScore] = dlAgent->rollout();

    /**
     * Run Deep Q-net. Variables for DQN agent
     */

    const auto ds_expo = std::make_shared<ExponentialDecayScheduler>(epsilonMin, epsilonMax, 0.00005);
    const float learning_rate = 1e-3f;
    int target_update_freq = 1000;
    size_t replay_capacity = 10000;
    float prepopulate_steps = 2500;
    size_t batch_size = 64;

    int state_size = env->reset().size();
    int action_size = env->getNumAction();
    std::vector hidden_layers = {128, 64, 63}; // You can change this

    
    // Create and train DQN agent
    std::vector<std::vector<float>> dqnRewards;

    std::unique_ptr<DQNAgent> dqnAagent;
    for (unsigned i = 0; i < 1; ++i) {
        std::vector<float> runRewards;
        dqnAagent = std::make_unique<DQNAgent>(env, state_size, action_size, hidden_layers, gamma, learning_rate,
            ds_expo, target_update_freq, replay_capacity,
            prepopulate_steps, batch_size);

        runRewards = dqnAagent->train(numEpisode);

        dqnRewards.push_back(runRewards);
    }
    auto [dqnSteps,dqnUtilScore] = dqnAagent->rollout();

    // Result
    //std::cout << "Random Policy: " << randSteps << " steps" << std::endl;
    std::cout << "Q Learning Policy: " << qlSteps << " steps | " << qlUScore << " Avg Utilization" << std::endl;
    std::cout << "Deep Q Network Policy: " << dqnSteps << " steps | " << dqnUtilScore << " Avg Utilization" << std::endl;

    Plot::ExportAverageRewardsOverEpisodes(qlRewards, "ql");
    //Plot::ExportAverageRewardsOverEpisodes(randRewards, "rand");
    Plot::ExportAverageRewardsOverEpisodes(dqnRewards, "dqn");

    return 0;
}
