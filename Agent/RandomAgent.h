#ifndef RANDOMAGENT_H
#define RANDOMAGENT_H

#include "BaseAgent.h"
#include "../Environment/Environment.h"

class RandomAgent : public BaseAgent {
public:
    RandomAgent(const std::shared_ptr<Environment> &env);
    RandomAgent(const std::shared_ptr<Environment> &env, unsigned seed);
    void update(std::vector<unsigned> s, unsigned a, int r, std::vector<unsigned> sPrime, bool done) override;
    unsigned getBehaviorPolicy(std::vector<unsigned> s, unsigned t) override;
    unsigned getTargetPolicy(std::vector<unsigned> s) override;
    std::tuple<std::vector<int>, unsigned> rollout(unsigned numEpisode);
private:
    std::shared_ptr<Environment> _env;
    std::mt19937 _randomizer;
};

#endif //RANDOMAGENT_H
