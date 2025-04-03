#include "Agent/QLAgent.h"
#include "Agent/DPAgent.h"
#include "Environment/Environment.h"

int main() {
    const std::shared_ptr<Environment> env = std::make_shared<Environment>(4, 25, 4, 10, 123, false);
    QLAgent agent = QLAgent(env, 0.5, 1, 0.1);
    DPAgent dp = DPAgent(env, 1, 0.01);

    agent.train(50000);
    agent.rollout();

    dp.run_value_iteration(); // train DP
    return 0;
}