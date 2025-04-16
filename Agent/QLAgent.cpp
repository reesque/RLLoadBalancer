#include "QLAgent.h"

#include "../Utils/ProgressBar.h"

QLAgent::QLAgent(const std::shared_ptr<Environment> &env, const float alpha, const float gamma,
                    const std::shared_ptr<DecayScheduler> &decayScheduler) {
    this->_env = env;
    this->_alpha = alpha;
    this->_gamma = gamma;
    this->_decayScheduler = decayScheduler;
    this->_randomizer = std::mt19937(std::random_device()());
}

QLAgent::QLAgent(const std::shared_ptr<Environment> &env, const float alpha, const float gamma,
                    const std::shared_ptr<DecayScheduler> &decayScheduler, const unsigned seed) {
    this->_env = env;
    this->_alpha = alpha;
    this->_gamma = gamma;
    this->_decayScheduler = decayScheduler;
    this->_randomizer = std::mt19937(seed);
}

unsigned QLAgent::getBehaviorPolicy(const std::vector<unsigned> s, const unsigned t) {
    auto randChance = std::uniform_real_distribution<float>(0, 1);

    float chance = randChance(this->_randomizer);
    if (chance < this->_decayScheduler->getValue(t)) {
        auto randAllAction = std::uniform_int_distribution<unsigned>(0, this->_env->getNumAction() - 1);
        return randAllAction(this->_randomizer);
    }

    return getTargetPolicy(s);
}

unsigned QLAgent::getTargetPolicy(const std::vector<unsigned> s) {
    auto sp = this->_q[this->_stateToKey(s)];
    return this->_argmax(sp);
}

void QLAgent::update(const std::vector<unsigned> s, const unsigned a, const float r, const std::vector<unsigned> sPrime, const bool done) {
    const unsigned bestAPrime = getTargetPolicy(sPrime);
    const auto nextQ = this->_q[this->_stateToKey(sPrime)][bestAPrime];
    const auto currentQ = this->_q[this->_stateToKey(s)][a];

    this->_q[this->_stateToKey(s)][a] += this->_alpha * (r + this->_gamma * nextQ - currentQ);
}

std::vector<float> QLAgent::train(const unsigned numEpisode) {
    this->_env->setDebug(false);
    std::vector<float> rewards = {};
    float uScore = 0.0;
    auto pb = ProgressBar("Training QL", numEpisode, [this, &rewards, &uScore](const unsigned it) {
        std::vector<unsigned> s = this->_env->reset();
        bool done = false;
        unsigned a = getBehaviorPolicy(s, it);
        float episodeRewards = 0;
        while (!done) {
            float r = 0;
            std::vector<unsigned> sPrime;
            std::tie(sPrime, r, done) = this->_env->step(a);
            const unsigned aPrime = getBehaviorPolicy(s, it);

            this->update(s, a, r, sPrime, done);

            episodeRewards += r;
            s = sPrime;
            a = aPrime;
        }

        rewards.push_back(episodeRewards);
    });

    return rewards;
}

std::tuple<unsigned, float> QLAgent::rollout() {
    std::vector<unsigned> s = this->_env->reset();
    this->_env->setDebug(true);
    bool done = false;
    unsigned a = getTargetPolicy(s);
    unsigned t = 0;
    while (!done) {
        float r = 0;
        std::vector<unsigned> sPrime;
        std::tie(sPrime, r, done) = this->_env->step(a);
        const unsigned aPrime = getTargetPolicy(s);

        s = sPrime;
        a = aPrime;
        ++t;
    }

    return std::make_tuple(t, this->_env->getUtilizationScore(t));
}

unsigned QLAgent::_argmax(const std::vector<float> &v) {
    float maxVal = v[0];
    for (int i = 1; i < v.size(); ++i) {
        if (maxVal < v[i]) {
            maxVal = v[i];
        }
    }

    std::vector<unsigned> maxIndices = {};

    for (int i = 0; i < v.size(); ++i) {
        if (v[i] == maxVal) {
            maxIndices.push_back(i);
        }
    }

    // Randomly choose among the max indices
    std::uniform_int_distribution<unsigned> dist(0, maxIndices.size() - 1);

    return maxIndices[dist(this->_randomizer)];
}

std::string QLAgent::_stateToKey(const std::vector<unsigned> &s) {
    std::stringstream ss;
    for (const unsigned v : s) {
        ss << v << ",";
    }

    if (!this->_q.contains(ss.str())) {
        this->_q[ss.str()] = {};
        for (int i = 0; i < this->_env->getNumAction(); i++) {
            this->_q[ss.str()].push_back(10.0);
        }
    }

    return ss.str();
}
