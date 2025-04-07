#include "DPAgent.h"

#include <sstream>

DPAgent::DPAgent(const std::shared_ptr<Environment> &environment, const float gamma, const float theta)
    : environment(environment), gamma(gamma), theta(theta) {

    const unsigned max_task = this->environment->getMaxDuration();
    const unsigned max_thread = this->environment->getMaxThread();
    const unsigned num_proc = this->environment->getNumProc();

    // Build all possible states
    for (unsigned t = 0; t <= max_task; ++t) {
        const std::vector init = {t};
        std::vector<std::vector<unsigned>> partial_states = {init};

        for (unsigned p = 0; p < num_proc; ++p) {
            std::vector<std::vector<unsigned>> new_states;
            for (const auto& state : partial_states) {
                for (unsigned bt = 0; bt <= max_thread; ++bt) {
                    std::vector<unsigned> extended = state;
                    extended.push_back(bt);
                    new_states.push_back(extended);
                }
            }
            partial_states = new_states;
        }

        all_states.insert(all_states.end(), partial_states.begin(), partial_states.end());
    }
}

std::string DPAgent::state_to_key(const std::vector<unsigned>& s) const {
    std::stringstream ss;
    for (unsigned v : s) {
        ss << v << ",";
    }
    return ss.str();
}

void DPAgent::update(std::vector<unsigned> s, const unsigned a, const int r, std::vector<unsigned> sPrime, const bool done) {
    const std::string skey = state_to_key(s);
    const float v_prime = done ? 0.0 : V[state_to_key(sPrime)];
    float q = static_cast<float>(r) + gamma * v_prime;

    if (q > V[skey]) {
        this->V[skey] = q;
        this->policy[skey] = a;
    }
}

void DPAgent::runValueIteration() {
    const unsigned num_actions = this->environment->getNumAction();

    reset();

    // Value Iteration Loop
    bool converged = false;
    while (!converged) {
        converged = true;
        for (const auto& s : all_states) {
            const float old_v = V[state_to_key(s)];
            V[state_to_key(s)] = -std::numeric_limits<double>::infinity();
            for (unsigned a = 0; a < num_actions; ++a) {
                auto [sPrime, r, done] = this->environment->simulateStep(s, a);
                this->update(s, a, r, sPrime, done);
            }
            float best_v = V[state_to_key(s)];

            if (std::abs(best_v - old_v) > theta) {
                converged = false;
            }
        }
    }
}

float DPAgent::rollout() {
    reset();

    bool done = false;
    std::vector<unsigned> s = this->environment->reset();
    unsigned a = getTargetPolicy(s);
    float totalReward = 0.0;
    float discount = 1.0;
    while (!done) {
        std::string skey = state_to_key(s);
        int r = 0;
        std::vector<unsigned> sPrime;
        std::tie(sPrime, r, done) = this->environment->simulateStep(s, a);
        const unsigned aPrime = getTargetPolicy(sPrime);

        totalReward += discount * static_cast<float>(r);
        discount *= gamma;
        s = sPrime;
        a = aPrime;
    }

    return totalReward;
}

void DPAgent::reset() {
    for (const auto& s : all_states) {
        this->V[state_to_key(s)] = 0.0;
    }
}

unsigned DPAgent::getTargetPolicy(std::vector<unsigned> s){
    return this->policy.at(state_to_key(s));
}

unsigned DPAgent::getBehaviorPolicy(std::vector<unsigned> s, unsigned t) {
    return getTargetPolicy(s);
}

