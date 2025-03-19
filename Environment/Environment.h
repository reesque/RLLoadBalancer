#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H
#include <deque>
#include <map>
#include <vector>
#include "Processor.h"
#include "Task.h"

class Environment {
public:
    explicit Environment(unsigned numProc, unsigned numTask, unsigned maxDuration, bool isDebug);
    explicit Environment(unsigned numProc, unsigned numTask, unsigned maxDuration, unsigned seed, bool isDebug);
    void reset();
    std::tuple<std::vector<std::unique_ptr<unsigned>>, int, bool> step(unsigned action);
    std::string toString() const;
    unsigned getNumAction() const;
    unsigned getNumProc() const;
    unsigned getNumTask() const;
    unsigned getMaxDuration() const;
private:
    unsigned _numProc;
    unsigned _numAction;
    unsigned _numTask;
    unsigned _maxDuration;
    unsigned _seed;
    bool _isDebug;
    std::map<unsigned, std::tuple<unsigned, unsigned>> _actionBetweenProcMap;
    std::deque<std::shared_ptr<Task>> _taskQueue;
    std::vector<std::shared_ptr<Processor>> _processors;

    void _moveBetweenProc(unsigned fromProc, unsigned toProc) const;
    void _moveFromQueue(unsigned toProc);
    void _generateActionMap();
};

#endif //ENVIRONMENT_H
