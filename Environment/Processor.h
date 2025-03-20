#ifndef PROCESSOR_H
#define PROCESSOR_H
#include <memory>
#include <vector>

#include "Task.h"

class Processor {
public:
    explicit Processor(unsigned maxThread);
    bool queue(const std::shared_ptr<Task> &task);
    unsigned getTotalProcessTime() const;
    float getUtilization() const;
    unsigned getNumBusyThread() const;
    void tick();
private:
    std::vector<std::shared_ptr<Task>> _tasks;
    unsigned _totalProcessTime = 0;
    unsigned _maxThread;
};

#endif //PROCESSOR_H
