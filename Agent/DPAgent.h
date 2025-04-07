#ifndef DPAGENT_H
#define DPAGENT_H

#include <unordered_map>
#include <vector>
#include <memory>

#include "BaseAgent.h"
#include "../Environment/Environment.h"

class DPAgent : public BaseAgent {
public:
    DPAgent(const std::shared_ptr<Environment> &environment, float gamma, float theta);
    void update(std::vector<unsigned> s, unsigned a, int r, std::vector<unsigned> sPrime, bool done) override;
    unsigned getTargetPolicy(std::vector<unsigned> s) override;
    unsigned getBehaviorPolicy(std::vector<unsigned> s, unsigned t) override;

    /**
     * Convert input state matrix to V-table string key
     */
    void runValueIteration();
    float rollout();
    void reset();
private:
    std::shared_ptr<Environment> environment;
    float gamma;
    float theta; // convergence threshold

    // key = stringified state, value = V(s)
    std::unordered_map<std::string, float> V;

    // key = stringified state, value = best action
    std::unordered_map<std::string, unsigned> policy;

    unsigned argmax(std::vector<float> v);
    std::string state_to_key(const std::vector<unsigned>& s) const;
    std::vector<std::vector<unsigned>> all_states;
};
    
#endif // DPAGENT_H