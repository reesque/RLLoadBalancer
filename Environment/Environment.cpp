#include "Environment.h"

#include <chrono>
#include <iostream>
#include <random>
#include <sstream>

Environment::Environment(const unsigned numProc, const unsigned numTask, const unsigned maxThread,
        const unsigned maxDuration, const unsigned seed, const bool isDebug) {
    this->_isDebug = isDebug;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_numTask = numTask;
    this->_seed = seed;
    this->_maxThread = maxThread;
    this->_maxDuration = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc + 1;

    this->reset();
}

Environment::Environment(const unsigned numProc, const unsigned numTask, const unsigned maxThread,
        const unsigned maxDuration, const bool isDebug) {
    this->_isDebug = isDebug;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_numTask = numTask;
    this->_seed = std::chrono::steady_clock::now().time_since_epoch().count();
    this->_maxThread = maxThread;
    this->_maxDuration = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc + 1;

    this->reset();
}

void Environment::reset() {
    this->_taskQueue.clear();
    this->_processors.clear();

    // Add processors
    for (int i = 0; i < this->_numProc; ++i) {
        this->_processors.push_back(std::make_shared<Processor>(this->_maxThread));
    }

    std::mt19937 rng = std::mt19937(this->_seed);

    for (int i = 0; i < this->_numTask; ++i) {
        std::uniform_int_distribution<unsigned> rand = std::uniform_int_distribution<unsigned>(1, this->_maxDuration);
        this->_taskQueue.push_back(std::make_shared<Task>(rand(rng)));
    }

    if (this->_isDebug) {
        std::cout << this->toString() << std::endl;
    }
}

std::tuple<std::vector<unsigned>, int, bool> Environment::step(const unsigned action) {
    // Action out of bound guard
    if (action >= this->_numAction) {
        std::stringstream es;
        es << "Given action out of range: " << action << std::endl;
        throw std::invalid_argument(es.str());
    }

    // Returning vars: nextState, reward, done
    std::vector<unsigned> nextState;
    int reward = 0;
    bool done = true;

    // Natural tick at the beginning
    for (const std::shared_ptr<Processor>& proc : this->_processors) {
        proc->tick();
    }

    // Perform action
    if (action < this->_numAction - 1) {
        // 0 -> (NumProc - 1): Move from queue to proc, with action being index of the proc to move to
        this->_moveFromQueue(action);
    }
    reward -= 1;

    // Check done
    nextState.push_back(this->_taskQueue.size());
    if (!this->_taskQueue.empty()) {
        done = false;
    }

    for (const std::shared_ptr<Processor>& proc : this->_processors) {
        nextState.push_back(proc->getNumBusyThread());
        if (proc->getNumBusyThread() != 0) {
            done = false;
        }

        if (this->_taskQueue.size() != 0 && proc->getUtilization() < 0.2) {
            reward -= 10;
        }
    }

    if (this->_isDebug) {
        std::cout << this->toString() << std::endl;
    }

    return std::make_tuple(nextState, reward, done);
}

unsigned Environment::getNumAction() const {
    return this->_numAction;
}

unsigned Environment::getNumTask() const {
    return this->_numTask;
}

unsigned Environment::getMaxThread() const {
    return this->_maxThread;
}

unsigned Environment::getNumProc() const {
    return this->_numProc;
}

std::string Environment::toString() const {
    std::stringstream result;

    result << "===========================================" << std::endl;
    result << "TASK QUEUE: [";
    for (const std::shared_ptr<Task>& task : this->_taskQueue) {
        result << task->getRemainingDuration() << " ";
    }
    result << "]" << std::endl;

    for (int i = 0; i < this->_processors.size(); ++i) {
        result << "PROCESS " << i << ": " << this->_processors.at(i)->getTotalProcessTime() << " STEPS" << std::endl;
    }
    result << "===========================================" << std::endl;

    return result.str();
}

void Environment::setDebug(const bool isDebug) {
    this->_isDebug = isDebug;
}

void Environment::_moveFromQueue(const unsigned toProc) {
    if (this->_taskQueue.empty()) {
        return;
    }

    const std::shared_ptr<Task> t = this->_taskQueue.front();
    this->_taskQueue.pop_front();

    // Processor is at max cap, re-queued back to the task queue
    if (!this->_processors.at(toProc)->queue(t)) {
        this->_taskQueue.push_front(t);
    }
}
