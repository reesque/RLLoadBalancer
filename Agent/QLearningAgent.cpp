#include "QLearningAgent.h"
#include <algorithm>
#include <random>

// Constructor and init Q_table
QLearningAgent::QLearningAgent(int states, int actions, double alpha, double gamma, double epsilon, std::shared_ptr<Environment> environment)
    : BaseAgent(environment), num_states(states), num_actions(actions), alpha(alpha), gamma(gamma), epsilon(epsilon) {
        this->Q_table = std::vector<std::vector<double>>(this->num_states, std::vector<double>(this->num_actions, 0.0));
};

int QLearningAgent::choose_action(int state) {
    // Random number generator
    std::random_device rd;   // seed is device
    std::mt19937 gen(rd());  // Mersenne Twister PRNG
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // decide whether behavior policy AND do exploration (e-soft)
    if (is_training && dis(gen) < epsilon) {
        std::uniform_int_distribution<int> action_dist(0, this->num_actions - 1);
        return action_dist(gen);
    } else { // or exploitation (greedy)
        double max_value = *std::max_element(this->Q_table[state].begin(), this->Q_table[state].end());

        // Collect all actions that have the max Q-value
        std::vector<int> best_actions; // All actions that have the max Q-value
        for (int i = 0; i < this->num_actions; i++) {
            if (this->Q_table[state][i] == max_value) {
                best_actions.push_back(i);
            }
        }

        // Random uniform Tie-breaker
        std::uniform_int_distribution<int> best_action_dist(0, best_actions.size() - 1);
        return best_actions[best_action_dist(gen)];
    }
}

void QLearningAgent::update(int state, int action, int next_state, double reward) {
    // Assumming terminal state will always be next_state, don't have to check for NULL next_state
    double best_next_action_value = *std::max_element(this->Q_table[next_state].begin(), this->Q_table[next_state].end()); // np.max(Q[s'])
    double td_target = reward + this->gamma * best_next_action_value;
    this->Q_table[state][action] += this->alpha * (td_target - this->Q_table[state][action]);
}

void QLearningAgent::train(unsigned episodes) {
    if (!this->env) {
        std::cerr << "ERROR: you forgot to set environment. Somehow." << std::endl;
        return;
    }

    // Ensure agent is in training mode
    train_mode();

    for (unsigned episode = 0; episode < episodes; ++episode) {
        this->env->reset();
        int state = 0;
        bool done = false;

        while (!done) {
            int action = choose_action(state);  // Now guaranteed to use ε-greedy

            std::vector<std::shared_ptr<unsigned int>> next_state_vector;
            int reward;
            bool is_done;
            std::tie(next_state_vector, reward, is_done) = this->env->step(action);

            int next_state = *(next_state_vector.front());  // Dereference shared_ptr<unsigned>
            update(state, action, next_state, reward);

            state = next_state;
            done = is_done;
        }
    }
}

// idk if I should keep this
void QLearningAgent::print_Q_table() const {
    std::cout << "Q_table:\n";
    for (const auto& row : this->Q_table) {
        for (double q_value : row) {
            std::cout << q_value << "\t";
        }
        std::cout << std::endl;
    }
}
    


