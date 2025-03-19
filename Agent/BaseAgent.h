#ifndef BASEAGENT_H
#define BASEAGENT_H

class BaseAgent {
public:
    virtual ~BaseAgent() = default;
    virtual void update() = 0;
    virtual void getBehaviorPolicy() = 0;
};

#endif //BASEAGENT_H
