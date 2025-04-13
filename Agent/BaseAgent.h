#ifndef BASEAGENT_H
#define BASEAGENT_H
#include <vector>

class BaseAgent {
public:
    virtual ~BaseAgent() = default;
    virtual void update(std::vector<unsigned> s, unsigned a, float r, std::vector<unsigned> sPrime, bool done) = 0;
    virtual unsigned getBehaviorPolicy(std::vector<unsigned> s, unsigned t) = 0;
    virtual unsigned getTargetPolicy(std::vector<unsigned> s) = 0;
};

#endif //BASEAGENT_H