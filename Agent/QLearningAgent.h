#ifndef QLEARNINGAGENT_H
#define QLEARNINGAGENT_H

#include <iostream>
#include <vector>
#include <memory>
#include "BaseAgent.h"
#include "Environment.h"

class QLearningAgent : public BaseAgent {
private:
    int num_states;
    int num_actions;
    double alpha;
    double gamma;
    double epsilon;
    std::vector<std::vector<double>> Q_table;

public:
    // Constructor now requires an Environment pointer
    QLearningAgent(int states, int actions, double alpha, double gamma, double epsilon, std::shared_ptr<Environment> environment);

    int choose_action(int state) override;
    void update(int state, int action, int next_state, double reward);
    void train(unsigned episodes);
    void print_Q_table() const; // idk if I should keep
};

#endif // QLEARNINGAGENT_H
