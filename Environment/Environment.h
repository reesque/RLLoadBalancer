#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H
#include <deque>
#include <vector>
#include "Processor.h"
#include "Task.h"

class Environment {
public:
    explicit Environment(unsigned numProc, unsigned numTask, unsigned maxThread, unsigned maxDuration, bool isDebug);
    explicit Environment(unsigned numProc, unsigned numTask, unsigned maxThread, unsigned maxDuration, unsigned seed, bool isDebug);
    void reset();
    std::tuple<std::vector<unsigned>, int, bool> step(unsigned action);
    std::string toString() const;
    unsigned getNumAction() const;
    unsigned getNumProc() const;
    unsigned getNumTask() const;
    unsigned getMaxThread() const;
    void setDebug(bool isDebug);
private:
    unsigned _numProc;
    unsigned _numAction;
    unsigned _numTask;
    unsigned _maxThread;
    unsigned _maxDuration;
    unsigned _seed;
    bool _isDebug;
    std::deque<std::shared_ptr<Task>> _taskQueue;
    std::vector<std::shared_ptr<Processor>> _processors;

    void _moveFromQueue(unsigned toProc);
};

#endif //ENVIRONMENT_H
