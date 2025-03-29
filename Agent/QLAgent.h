#ifndef QLAGENT_H
#define QLAGENT_H
#include <random>
#include <torch/torch.h>
#include "BaseAgent.h"
#include "../Environment/Environment.h"

class QLAgent: BaseAgent {
public:
    explicit QLAgent(const std::shared_ptr<Environment> &env, float alpha, float gamma, float epsilon);
    explicit QLAgent(const std::shared_ptr<Environment> &env, float alpha, float gamma, float epsilon, unsigned seed);
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
    std::mt19937 _randomizer;
    torch::Tensor _q;

    unsigned _argmax(const torch::Tensor& v);
    static std::vector<at::indexing::TensorIndex> _getIndicesTensor(std::vector<unsigned> s);
    static std::vector<at::indexing::TensorIndex> _getIndicesTensor(std::vector<unsigned> s, unsigned a);
};

#endif //QLAGENT_H
