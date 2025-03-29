#include "Agent/QLAgent.h"
#include "Environment/Environment.h"

int main() {
    const auto env = std::make_shared<Environment>(2, 20, 4, 10, 123);
    auto agent = QLAgent(env, 0.5, 1, 0.1, 123);

    agent.train(100000);
    agent.rollout();

    return 0;
}