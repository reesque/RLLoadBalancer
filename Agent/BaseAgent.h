#ifndef BASEAGENT_H
#define BASEAGENT_H
#include <vector>

#include <memory>
#include "../Environment/Environment.h"

class BaseAgent {
protected:
    bool is_training = true; // Default to training mode
    std::shared_ptr<Environment> env; // Pointer to Environment
public:
    explicit BaseAgent(std::shared_ptr<Environment> environment) : env(environment) {}
    virtual ~BaseAgent() = default;
    virtual void update(std::vector<unsigned> s, unsigned a, int r, std::vector<unsigned> sPrime) = 0;
    virtual unsigned getBehaviorPolicy(std::vector<unsigned> s) = 0;
    virtual unsigned getTargetPolicy(std::vector<unsigned> s) = 0;
};

#endif //BASEAGENT_H
