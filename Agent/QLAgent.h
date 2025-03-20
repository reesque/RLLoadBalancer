#ifndef QLAGENT_H
#define QLAGENT_H
#include "BaseAgent.h"
#include "../Environment/Environment.h"

class QLAgent: BaseAgent {
public:
    explicit QLAgent(const std::shared_ptr<Environment> &env, float alpha, float gamma, float epsilon);
    unsigned getBehaviorPolicy(std::vector<unsigned> s) override;
    unsigned getTargetPolicy(std::vector<unsigned> s) override;
    void update(std::vector<unsigned> s, unsigned a, int r, std::vector<unsigned> sPrime) override;
    void train(unsigned numRun);
    void rollout();
private:
    std::shared_ptr<Environment> _env;
    float _alpha;
    float _gamma;
    float _epsilon;
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>> _q;

    unsigned _argmax(std::vector<float> v);
};

#endif //QLAGENT_H
