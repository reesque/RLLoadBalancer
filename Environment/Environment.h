#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H
#include <deque>
#include <random>
#include <vector>
#include "Processor.h"
#include "Task.h"

class Environment {
public:
    explicit Environment(unsigned numProc, unsigned maxThread, unsigned maxDuration, unsigned numTask);
    explicit Environment(unsigned numProc, unsigned maxThread, unsigned maxDuration, unsigned numTask, unsigned seed);

    void generateTasks();
    std::vector<unsigned> reset();
    std::tuple<std::vector<unsigned>, int, bool> step(unsigned action);
    std::string toString() const;
    unsigned getNumAction() const;
    unsigned getNumProc() const;
    unsigned getMaxThread() const;
    unsigned getMaxDuration() const;
    void setDebug(bool isDebug);
private:
    unsigned _numProc;
    unsigned _numAction;
    unsigned _maxThread;
    unsigned _maxDuration;
    unsigned _numTask;
    std::mt19937 _randomizer;
    bool _isDebug;
    std::deque<std::shared_ptr<Task>> _taskQueue;
    std::deque<std::shared_ptr<Task>> _initialTaskQueue;
    std::vector<std::shared_ptr<Processor>> _processors;

    void _moveFromQueue(unsigned toProc);
};

#endif //ENVIRONMENT_H
