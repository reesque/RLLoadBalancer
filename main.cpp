#include "Agent/QLAgent.h"
#include "Agent/QLearningAgent.h"
#include "Environment/Environment.h"

int main() {
    const std::shared_ptr<Environment> env = std::make_shared<Environment>(4, 25, 4, 10, 123, false);
    // QLAgent agent = QLAgent(env, 0.5, 1, 0.1);
    QLearningAgent agent = QLearningAgent(0.5, 1, 0.1, env);
    

    agent.train(50000);
    agent.rollout();

    return 0;
}