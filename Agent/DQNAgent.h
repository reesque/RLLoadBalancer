#ifndef DQN_AGENT_H
#define DQN_AGENT_H

#include <memory>
#include <vector>
#include <random>
#include <torch/torch.h>

#include "BaseAgent.h"
#include "FFN.h"
#include "ReplayBuffer.h"
#include "../Environment/Environment.h"
#include "../DecayScheduler/DecayScheduler.h"


class DQNAgent: BaseAgent {
public:
    DQNAgent(
        const std::shared_ptr<Environment> &env, 
        int state_size, 
        int action_size, 
        const std::vector<int>& hidden_layers,
        float gamma,
        float lr,
        const std::shared_ptr<DecayScheduler> &decayScheduler,
        int target_update_freq, // 10K
        size_t replay_size,
        float replay_prepopulate_steps,
        size_t batch_size
    );
    unsigned getBehaviorPolicy(std::vector<unsigned> s, unsigned t) override;
    unsigned getTargetPolicy(std::vector<unsigned> s) override;
    void update(std::vector<unsigned> s, unsigned a, float r, std::vector<unsigned> sPrime, bool done) override;
    std::vector<float> train(unsigned numEpisode);
    unsigned rollout();

private:
    std::shared_ptr<Environment> _env;
    std::shared_ptr<DecayScheduler> _decay_scheduler;

    FFN _q_net = nullptr;
    FFN _target_net = nullptr;
    torch::optim::Adam _optimizer;
    std::shared_ptr<ReplayBuffer> _replay_buffer;
    std::mt19937 _randomizer;

    int _state_size;
    int _action_size;
    float _gamma;
    int _target_update_freq;
    int _steps_done = 0;
    size_t _batch_size;

    /**
     * @brief Performs one training step of the DQN using a minibatch from the replay buffer.
     *
     * This method implements the core learning algorithm of DQN:
     * - Samples a minibatch of transitions from the replay buffer.
     * - Computes Q(s, a) from the current Q-network.
     * - Computes target Q-values using the target network and the Bellman equation.
     * - Calculates the mean squared error (MSE) loss.
     * - Performs a gradient descent step to minimize the loss.
     *
     * If there are not enough samples in the replay buffer, this function exits early.
     */
    void _trainStep(); // helper for learning from replay buffer

    /**
     * Updates the target network by copying weights from the Q-network.
     */
    void _updateTargetNetwork(); // helper to update target network
    unsigned _argmax(const torch::Tensor& v);
};

#endif // DQN_AGENT_H