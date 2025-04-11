#include <iostream>

#include "Agent/QLAgent.h"
#include "Agent/RandomAgent.h"
#include "DecayScheduler/LinearDecayScheduler.h"
#include "DecayScheduler/ExponentialDecayScheduler.h"
#include "Environment/Environment.h"
#include "Utils/Plot.h"

#include <numeric>

int main() {
    const unsigned seed = 123;
    const unsigned numProc = 2;
    const unsigned numThread = 2;
    const unsigned numTask = 20;
    const unsigned maxTaskDuration = 100;

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
    const auto env = std::make_shared<Environment>(numProc, numThread, maxTaskDuration, numTask, seed);
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
    std::unique_ptr<QLAgent> agent;
    for (unsigned i = 0; i < numRun; i++) {
        agent = std::make_unique<QLAgent>(env, alpha, gamma, ds, seed);
        rewards.push_back(agent->train(numEpisode));
    }
    */
    // Run Deep Q-net
    /**
     * Variables for DQN agent
     */
    const auto ds_expo = std::make_shared<ExponentialDecayScheduler>(epsilonMin, epsilonMax, epsilonDecayRate);
    const float learning_rate = 1e-3f;
    int target_update_freq = 1000;
    size_t replay_capacity = 10000;
    float prepopulate_steps = 2500;
    size_t batch_size = 64;

    int state_size = env->reset().size();
    int action_size = env->getNumAction();
    std::vector<int> hidden_layers = {64, 64}; // You can change this

    
    // // Create and train DQN agent
    std::unique_ptr<DQNAgent> agent;
    for (unsigned i = 0; i < numRun; i++) {
        agent = std::make_unique<DQNAgent>(env, state_size, action_size, hidden_layers, gamma, learning_rate,
            ds_expo, target_update_freq, replay_capacity,
            prepopulate_steps, batch_size);
        rewards.push_back(agent->train(numEpisode));
    }

    const unsigned qlSteps = dlAgent->rollout();

    // Result
    std::cout << "Random Policy: " << randSteps << " steps" << std::endl;
    std::cout << "Q Learning Policy: " << qlSteps << " steps" << std::endl;

    Plot::ExportAverageRewardsOverEpisodes(qlRewards, "ql");
    Plot::ExportAverageRewardsOverEpisodes(randRewards, "rand");
    return 0;
}
