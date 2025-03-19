#include "Environment.h"

#include <chrono>
#include <iostream>
#include <random>
#include <sstream>

Environment::Environment(const unsigned numProc, const unsigned numTask, const unsigned maxDuration, const unsigned seed, const bool isDebug) {
    this->_isDebug = isDebug;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_numTask = numTask;
    this->_seed = seed;
    this->_maxDuration = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc + (numProc * (numProc - 1)) + 1;

    this->reset();
}

Environment::Environment(const unsigned numProc, const unsigned numTask, const unsigned maxDuration, const bool isDebug) {
    this->_isDebug = isDebug;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_numTask = numTask;
    this->_seed = std::chrono::steady_clock::now().time_since_epoch().count();
    this->_maxDuration = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc + (numProc * (numProc - 1)) + 1;

    this->reset();
}

void Environment::reset() {
    this->_taskQueue.clear();
    this->_processors.clear();

    // Add processors
    for (int i = 0; i < this->_numProc; i++) {
        this->_processors.push_back(std::make_shared<Processor>(this->_maxDuration));
    }

    std::mt19937 rng = std::mt19937(this->_seed);

    for (int i = 0; i < this->_numTask; i++) {
        std::uniform_int_distribution<unsigned> rand = std::uniform_int_distribution<unsigned>(1, this->_maxDuration);
        this->_taskQueue.push_back(std::make_shared<Task>(rand(rng)));
    }

    if (this->_isDebug) {
        std::cout << this->toString() << std::endl;
    }
}

std::tuple<std::vector<int>, int, bool> Environment::step(const int action) {
    // Returning vars: nextState, reward, done
    std::vector<int> nextState;
    int reward = 0;
    bool done = true;

    // Natural tick at the beginning
    for (const std::shared_ptr<Processor>& proc : this->_processors) {
        proc->tick();
    }

    // Perform action
    if (action == this->_numAction - 1) {
        // MaxAction - 1: Do nothing action
    } else if (action >= 0 && action < this->_numProc) {
        // 0 -> (NumProc - 1): Move from queue to proc, with action being index of the proc to move to
        this->_moveFromQueue(action);
    } else {
        // (NumProc - 1) -> (MaxAction - 2): Move from one proc to another
        const int fromProc = (action - this->_numAction) / (this->_numProc - 1);
        const int toProc = (action - this->_numAction) % (this->_numProc - 1);
        this->_moveBetweenProc(fromProc, toProc);
    }

    // Check done
    for (const std::shared_ptr<Processor>& proc : this->_processors) {
        nextState.push_back(proc->getTotalProcessTime());
        if (proc->getTotalProcessTime() != 0) {
            done = false;
            reward = -1;
        }
    }

    nextState.push_back(this->_taskQueue.size());
    if (!this->_taskQueue.empty()) {
        done = false;
        reward = -1;
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

unsigned Environment::getMaxDuration() const {
    return this->_maxDuration;
}

unsigned Environment::getNumProc() const {
    return this->_numProc;
}

std::string Environment::toString() const {
    std::stringstream result;

    result << "===========================================" << std::endl;
    result << "TASK QUEUE: [";
    for (int i = 0; i < this->_taskQueue.size(); i++) {
        result << this->_taskQueue.at(i)->getRemainingDuration() << " ";
    }
    result << "]" << std::endl;

    for (int i = 0; i < this->_processors.size(); i++) {
        result << "PROCESS " << i << ": " << this->_processors.at(i)->getTotalProcessTime() << " STEPS" << std::endl;
    }
    result << "===========================================" << std::endl;

    return result.str();
}

void Environment::_moveBetweenProc(const unsigned fromProc, const unsigned toProc) const {
    const std::shared_ptr<Task> taskToMove = this->_processors.at(fromProc)->getLastInQueue();

    // Processor is at max cap, re-queued back to the previous proc
    if (!this->_processors.at(toProc)->queue(taskToMove)) {
        this->_processors.at(fromProc)->queue(taskToMove);
    }
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
