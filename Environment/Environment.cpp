#include "Environment.h"

#include <iostream>
#include <sstream>

Environment::Environment(const unsigned numProc, const unsigned maxThread, const unsigned maxDuration, const unsigned numTask, const unsigned seed) {
    this->_isDebug = false;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_randomizer = std::mt19937(seed);
    this->_maxThread = maxThread;
    this->_maxDuration = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc;
    this->_numTask = numTask;

    this->generateTasks();
    this->reset();
}

Environment::Environment(const unsigned numProc, const unsigned maxThread, const unsigned maxDuration, const unsigned numTask) {
    this->_isDebug = false;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_randomizer = std::mt19937(std::random_device()());
    this->_maxThread = maxThread;
    this->_maxDuration = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc;
    this->_numTask = numTask;

    this->generateTasks();
    this->reset();
}

void Environment::generateTasks() {
    for (unsigned i = 0; i < this->_numTask; i++) {
        auto rand = std::uniform_int_distribution<unsigned>(1, this->_maxDuration);
        unsigned newLength = rand(this->_randomizer);
        this->_initialTaskQueue.push_back(std::make_shared<Task>(newLength));
    }
}

std::vector<unsigned> Environment::reset() {
    std::vector<unsigned> s = {};
    this->_taskQueue.clear();
    this->_processors.clear();

    // Add tasks
    for (int i = 0; i < this->_initialTaskQueue.size(); ++i) {
        this->_taskQueue.push_back(std::make_shared<Task>(_initialTaskQueue[i]->getRemainingDuration()));
    }
    s.push_back(this->_taskQueue.at(0)->getRemainingDuration());

    // Add processors
    for (int i = 0; i < this->_numProc; ++i) {
        this->_processors.push_back(std::make_shared<Processor>(this->_maxThread));
        for (int j = 0; j < this->_maxThread; ++j) {
            s.push_back(0);
        }
    }

    if (this->_isDebug) {
        std::cout << this->toString() << std::endl;
    }

    return s;
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
    this->_moveFromQueue(action);
    reward -= 1;

    // Check done
    if (!this->_taskQueue.empty()) {
        done = false;
        nextState.push_back(this->_taskQueue.at(0)->getRemainingDuration());
    } else {
        nextState.push_back(0);
    }

    for (const std::shared_ptr<Processor>& proc : this->_processors) {
        auto threadsLength = proc->getThreadsLength();
        for (unsigned length : threadsLength) {
            nextState.push_back(length);
        }

        if (proc->getNumBusyThread() != 0) {
            done = false;
        }

        if (!this->_taskQueue.empty() && proc->getUtilization() < 0.2) {
            reward -= 1;
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

unsigned Environment::getMaxThread() const {
    return this->_maxThread;
}

unsigned Environment::getNumProc() const {
    return this->_numProc;
}

unsigned Environment::getMaxDuration() const {
    return this->_maxDuration;
}

float Environment::getUtilizationScore() const {
    float mean = 0;
    for (const auto & _processor : this->_processors) {
        mean += _processor->getUtilization();
    }

    mean /= static_cast<float>(this->getNumProc());

    float variance = 0.0;
    for (const auto & _processor : this->_processors) {
        variance += (_processor->getUtilization() - mean) * (_processor->getUtilization() - mean);
    }
    variance /= static_cast<float>(this->getNumProc());
    const float stddev = std::sqrt(variance);

    const float score = 1.0f - (stddev / mean);

    return std::max(0.0f, score);
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
