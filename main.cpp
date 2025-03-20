#include "Agent/QLAgent.h"
#include "Environment/Environment.h"

int main() {
    const std::shared_ptr<Environment> env = std::make_shared<Environment>(4, 7, 14, 123, false);
    QLAgent agent = QLAgent(env, 0.5, 1, 0.1);

    agent.train(50000);
    agent.rollout();

    return 0;
}