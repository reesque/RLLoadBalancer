#ifndef QLEARNINGAGENT_H
#define QLEARNINGAGENT_H

#include "BaseAgent.h"
#include "../Environment/Environment.h"

class QLearningAgent : public BaseAgent {
private:
    double alpha;
    double gamma;
    double epsilon;
    std::shared_ptr<Environment> environment;
    std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>> Q_table;

    unsigned argmax(std::vector<float> v);

public:
    // Constructor now requires an Environment pointer
    explicit QLearningAgent(double alpha, double gamma, double epsilon, const std::shared_ptr<Environment> &environment);

    unsigned getBehaviorPolicy(std::vector<unsigned> s) override;
    unsigned getTargetPolicy(std::vector<unsigned> s) override;
    void update(std::vector<unsigned> s, unsigned a, int r, std::vector<unsigned> sPrime) override;
    void train(unsigned episodes);
    void rollout();
};

#endif // QLEARNINGAGENT_H
