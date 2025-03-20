#include "QLAgent.h"

#include <algorithm>
#include <iostream>
#include <random>

QLAgent::QLAgent(const std::shared_ptr<Environment> &env, const float alpha, const float gamma, const float epsilon) {
    this->_env = env;
    this->_alpha = alpha;
    this->_gamma = gamma;
    this->_epsilon = epsilon;

    this->_q = std::vector(env->getNumTask() + 1, std::vector(env->getMaxThread() + 1,std::vector(
        env->getMaxThread() + 1, std::vector(env->getMaxThread() + 1, std::vector(
            env->getMaxThread() + 1,std::vector(env->getNumAction(), 10.0f))))));
}

unsigned QLAgent::getBehaviorPolicy(const std::vector<unsigned> s) {
    std::random_device rd;
    std::mt19937 rng = std::mt19937(rd());
    std::uniform_real_distribution<float> randChance = std::uniform_real_distribution<float>(0, 1);

    float chance = randChance(rng);
    if (chance < this->_epsilon) {
        std::uniform_real_distribution<float> randAllAction = std::uniform_real_distribution<float>(0, this->_env->getNumAction());
        return randAllAction(rng);
    }

    std::vector qs = this->_q[s[0]][s[1]][s[2]][s[3]][s[4]];
    return _argmax(qs);
}

unsigned QLAgent::getTargetPolicy(const std::vector<unsigned> s) {
    std::vector qs = this->_q[s[0]][s[1]][s[2]][s[3]][s[4]];
    return _argmax(qs);
}

void QLAgent::update(const std::vector<unsigned> s, const unsigned a, const int r, const std::vector<unsigned> sPrime) {
    std::vector qsPrime = this->_q[sPrime[0]][sPrime[1]][sPrime[2]][sPrime[3]][sPrime[4]];
    unsigned bestAPrime = _argmax(qsPrime);

    this->_q[s[0]][s[1]][s[2]][s[3]][s[4]][a] += this->_alpha *
        (r + this->_gamma * this->_q[sPrime[0]][sPrime[1]][sPrime[2]][sPrime[3]][sPrime[4]][bestAPrime] - this->_q[s[0]][s[1]][s[2]][s[3]][s[4]][a]);
}

void QLAgent::train(const unsigned numRun) {
    this->_env->setDebug(false);
    for (unsigned i = 0; i < numRun; i++) {
        std::cout << "Training Run " << i << std::endl;
        this->_env->reset();
        std::vector<unsigned> s = {this->_env->getNumTask(), 0, 0, 0, 0};
        bool done = false;
        unsigned a = getBehaviorPolicy(s);
        while (!done) {
            int r = 0;
            std::vector<unsigned> sPrime;
            std::tie(sPrime, r, done) = this->_env->step(a);
            const unsigned aPrime = getBehaviorPolicy(s);

            this->update(s, a, r, sPrime);

            s = sPrime;
            a = aPrime;
        }
    }
}

void QLAgent::rollout() {
    this->_env->setDebug(true);
    this->_env->reset();
    std::vector<unsigned> s = {this->_env->getNumTask(), 0, 0, 0, 0};
    bool done = false;
    unsigned a = getTargetPolicy(s);
    unsigned t = 0;
    while (!done) {
        int r = 0;
        std::vector<unsigned> sPrime;
        std::tie(sPrime, r, done) = this->_env->step(a);
        const unsigned aPrime = getTargetPolicy(s);

        this->update(s, a, r, sPrime);

        s = sPrime;
        a = aPrime;
        ++t;
    }
    std::cout << "Took " << t << " time steps to finish!" << std::endl;
}

unsigned QLAgent::_argmax(std::vector<float> v) {
    if (v.empty()) return -1; // Handle empty input

    const double max_value = *std::ranges::max_element(v);
    std::vector<int> maxIndices;

    // Collect all indices with the max value
    for (int i = 0; i < v.size(); ++i) {
        if (v[i] == max_value) {
            maxIndices.push_back(i);
        }
    }

    // Randomly choose among the max indices
    std::random_device rd;
    std::mt19937 rng = std::mt19937(rd());
    std::uniform_int_distribution<int> dist(0, maxIndices.size() - 1);

    return maxIndices[dist(rng)];
}

