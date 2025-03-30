#include "QLearningAgent.h"
#include <iostream>
#include <algorithm>
#include <random>

// Constructor and init Q_table
QLearningAgent::QLearningAgent(double alpha, double gamma, double epsilon, const std::shared_ptr<Environment> &environment)
    : environment(environment), alpha(alpha), gamma(gamma), epsilon(epsilon) {
        this->Q_table = std::vector(environment->getNumTask() + 1, std::vector(environment->getMaxThread() + 1,std::vector(
            environment->getMaxThread() + 1, std::vector(environment->getMaxThread() + 1, std::vector(
                environment->getMaxThread() + 1,std::vector(environment->getNumAction(), 10.0f))))));
};

unsigned QLearningAgent::argmax(std::vector<float> v) {
    double max_value = *std::max_element(v.begin(), v.end());

    // Collect all actions that have the max Q-value
    std::vector<int> best_actions; // All actions that have the max Q-value
    for (int i = 0; i < v.size(); i++) {
        if (v[i] == max_value) {
            best_actions.push_back(i);
        }
    }

    // Random uniform Tie-breaker
    std::random_device rd;   // seed
    std::mt19937 gen(rd());  // Mersenne Twister PRNG
    std::uniform_int_distribution<int> best_action_dist(0, best_actions.size() - 1);
    return best_actions[best_action_dist(gen)];
}


unsigned QLearningAgent::getBehaviorPolicy(std::vector<unsigned> s) {
    // Random number generator
    std::random_device rd;   // seed
    std::mt19937 gen(rd());  // Mersenne Twister PRNG
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    // decide whether behavior policy AND do exploration (e-soft)
    if (dis(gen) < this->epsilon) {
        std::uniform_int_distribution<int> action_dist(0, this->environment->getNumAction() - 1);
        return action_dist(gen);
    } else { // or exploitation (greedy)
       return getTargetPolicy(s);
    }
}

unsigned QLearningAgent::getTargetPolicy(std::vector<unsigned> s) {
    std::vector<float> v = this->Q_table[s[0]][s[1]][s[2]][s[3]][s[4]];
    return argmax(v);
}

void QLearningAgent::update(std::vector<unsigned> s, unsigned a, int r, std::vector<unsigned> sPrime) {
    // Assumming terminal state will always be next_state, don't have to check for NULL next_state
    std::vector<float> QSPrime_v = this->Q_table[sPrime[0]][sPrime[1]][sPrime[2]][sPrime[3]][sPrime[4]];
    double best_next_action_value = *std::max_element(QSPrime_v.begin(), QSPrime_v.end()); // np.max(Q[s'])
    double td_target = r + this->gamma * best_next_action_value;
    this->Q_table[s[0]][s[1]][s[2]][s[3]][s[4]][a] += this->alpha * (td_target - this->Q_table[s[0]][s[1]][s[2]][s[3]][s[4]][a]);
}

void QLearningAgent::train(unsigned episodes) {
    if (!this->environment) {
        std::cerr << "ERROR: you forgot to set environment. Somehow." << std::endl;
        return;
    }

    for (unsigned episode = 0; episode < episodes; ++episode) {
        this->environment->setDebug(false);
        this->environment->reset();

        std::vector<unsigned> state = {this->environment->getNumTask(), 0, 0, 0, 0};
        bool done = false;
        // std::cout << "Episode " << episode << std::endl;

        while (!done) {
            
            int action = getBehaviorPolicy(state); 

            std::vector<unsigned> next_state;
            int reward;
            bool is_done;
            std::tie(next_state, reward, is_done) = this->environment->step(action);

            update(state, action, reward, next_state);

            state = next_state;
            done = is_done;
        }
        // std::cout << "Done!" << std::endl;
    }
}

void QLearningAgent::rollout() {
    //Borrowed
    this->environment->setDebug(true);
    this->environment->reset();
    std::vector<unsigned> state = {this->environment->getNumTask(), 0, 0, 0, 0};
    bool done = false;
    unsigned t = 0;
    while (!done) {
        int action = getTargetPolicy(state);  // Now guaranteed to use ε-greedy

        std::vector<unsigned> next_state;
        int reward;
        bool is_done;
        std::tie(next_state, reward, is_done) = this->environment->step(action);

        update(state, action, reward, next_state);

        state = next_state;
        done = is_done;
        ++t;
    }
    std::cout << "Done!" << std::endl;
    std::cout << "Took " << t << " time steps to finish!" << std::endl;
}