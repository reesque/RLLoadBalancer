#ifndef QLAGENT_H
#define QLAGENT_H
#include <random>
#include <unordered_map>

#include "BaseAgent.h"
#include "../DecayScheduler/DecayScheduler.h"
#include "../Environment/Environment.h"

class QLAgent : public BaseAgent {
public:
    QLAgent(const std::shared_ptr<Environment> &env, float alpha, float gamma, const std::shared_ptr<DecayScheduler> &decayScheduler);
    QLAgent(const std::shared_ptr<Environment> &env, float alpha, float gamma, const std::shared_ptr<DecayScheduler> &decayScheduler, unsigned seed);
    unsigned getBehaviorPolicy(std::vector<unsigned> s, unsigned t) override;
    unsigned getTargetPolicy(std::vector<unsigned> s) override;
    void update(std::vector<unsigned> s, unsigned a, float r, std::vector<unsigned> sPrime, bool done) override;
    std::vector<float> train(unsigned numEpisode);
    std::tuple<unsigned, float> rollout();
private:
    std::shared_ptr<Environment> _env;
    float _alpha;
    float _gamma;
    std::shared_ptr<DecayScheduler> _decayScheduler;
    std::mt19937 _randomizer;
    std::unordered_map<std::string, std::vector<float>> _q;

    unsigned _argmax(const std::vector<float> &v);
    std::string _stateToKey(const std::vector<unsigned> &s);
};

#endif //QLAGENT_H
