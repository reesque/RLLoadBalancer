#include "RandomAgent.h"

#include <iostream>

#include "../Utils/ProgressBar.h"

RandomAgent::RandomAgent(const std::shared_ptr<Environment> &env) {
    this->_env = env;
    this->_randomizer = std::mt19937(std::random_device()());
}

RandomAgent::RandomAgent(const std::shared_ptr<Environment> &env, const unsigned seed) {
    this->_env = env;
    this->_randomizer = std::mt19937(seed);
}

void RandomAgent::update(const std::vector<unsigned> s, const unsigned a, const float r,
    const std::vector<unsigned> sPrime, const bool done) {}

unsigned RandomAgent::getBehaviorPolicy(const std::vector<unsigned> s, const unsigned t) {
    std::uniform_int_distribution<unsigned> dist(0, this->_env->getNumAction() - 1);

    return dist(this->_randomizer);
}

unsigned RandomAgent::getTargetPolicy(const std::vector<unsigned> s) {return 0;}


std::tuple<std::vector<float>, unsigned> RandomAgent::rollout(const unsigned numEpisode) {
    this->_env->setDebug(false);
    std::vector<float> rewards = {};
    unsigned totalSteps = 0;
    auto pb = ProgressBar("Rollout Random", numEpisode, [this, &rewards, &totalSteps](const unsigned it) {
        std::vector<unsigned> s = this->_env->reset();
        bool done = false;
        unsigned a = getBehaviorPolicy(s, it);
        int episodeRewards = 0;
        unsigned step = 0;
        while (!done) {
            int r = 0;
            std::vector<unsigned> sPrime;
            std::tie(sPrime, r, done) = this->_env->step(a);
            const unsigned aPrime = getBehaviorPolicy(s, it);

            this->update(s, a, r, sPrime, done);

            step += 1;
            episodeRewards += r;
            s = sPrime;
            a = aPrime;
        }

        totalSteps += step;
        rewards.push_back(episodeRewards);
    });

    unsigned avgSteps = totalSteps / numEpisode;

    return std::make_tuple(rewards, avgSteps);
}
