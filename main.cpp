#include "Agent/QLAgent.h"
#include "DecayScheduler/LinearDecayScheduler.h"
#include "Environment/Environment.h"

int main() {
    const auto env = std::make_shared<Environment>(2, 20, 4, 10, 123);
    const auto ds = std::make_shared<LinearDecayScheduler>(0.1, 1.0, 0.001);
    auto agent = QLAgent(env, 0.5, 1, ds, 123);

    agent.train(100000);
    agent.rollout();

    return 0;
}