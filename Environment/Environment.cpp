#include "Environment.h"

#include <iostream>
#include <sstream>

Environment::Environment(const unsigned numProc, const unsigned maxThread, const unsigned maxDuration, const unsigned seed) {
    this->_isDebug = false;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_randomizer = std::mt19937(seed);
    this->_maxThread = maxThread;
    this->_maxDuration = maxDuration;
    this->_remainingDurationInQueue = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc + 1;

    this->generateTasks();
    this->reset();
}

Environment::Environment(const unsigned numProc, const unsigned maxThread, const unsigned maxDuration) {
    this->_isDebug = false;
    this->_processors = std::vector<std::shared_ptr<Processor>>();
    this->_randomizer = std::mt19937(std::random_device()());
    this->_maxThread = maxThread;
    this->_maxDuration = maxDuration;
    this->_remainingDurationInQueue = maxDuration;
    this->_numProc = numProc;
    this->_numAction = numProc + 1;

    this->generateTasks();
    this->reset();
}

void Environment::generateTasks() {
    unsigned remainingLength = this->_maxDuration;
    const unsigned maxTaskLength = std::max(1u, this->_maxDuration / 4);  // Cap each task's max length

    while (remainingLength > 0) {
        const unsigned upperBound = std::min(maxTaskLength, remainingLength);
        auto rand = std::uniform_int_distribution<unsigned>(1, upperBound);
        unsigned newLength = rand(this->_randomizer);
        this->_initialTaskQueue.push_back(std::make_shared<Task>(newLength));

        remainingLength -= newLength;
    }
}

std::vector<unsigned> Environment::reset() {
    std::vector<unsigned> s = {};
    this->_taskQueue.clear();
    this->_processors.clear();

    // Add tasks
    this->_remainingDurationInQueue = this->_maxDuration;
    for (int i = 0; i < this->_initialTaskQueue.size(); ++i) {
        this->_taskQueue.push_back(std::make_shared<Task>(_initialTaskQueue[i]->getRemainingDuration()));
    }
    s.push_back(this->_remainingDurationInQueue);

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
    if (action < this->_numAction - 1) {
        // 0 -> (NumProc - 1): Move from queue to proc, with action being index of the proc to move to
        this->_moveFromQueue(action);
    }
    reward -= 1;

    // Check done
    nextState.push_back(this->_remainingDurationInQueue);
    if (!this->_taskQueue.empty()) {
        done = false;
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

std::tuple<std::vector<unsigned>, int, bool> Environment::simulateStep(const std::vector<unsigned>& state, const unsigned action) const {
    /*
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

    // Pseudo-Processors for simulation
    const unsigned num_proc = this->getNumProc();
    const unsigned max_threads = this->getMaxThread();
    unsigned num_tasks = state[0];  // task queue size

    // Simulate tick

    for (unsigned proc = 0; proc < num_proc; ++proc) {
        std::vector proc_queue(state.begin() + 1 + max_threads * proc, state.begin() + max_threads + max_threads * proc);
        bool actionExecuted = false;
        for (const unsigned& length : proc_queue) {
            if (num_tasks > 0 && action == proc && !actionExecuted && length - 1 != 0) {
                nextState.push_back(std::max(static_cast<int>(length) - 1, 0));
            }
            nextState.push_back(std::max(static_cast<int>(length) - 1, 0));
        }
    }

    nextState.insert(nextState.begin(), std::max(static_cast<int>(num_tasks) - 1, 0));

    // Perform action
    if (action < num_proc && num_tasks > 0 && proc_threads[action] < max_threads) {
        proc_threads[action] += 1;
        num_tasks -= 1;
    }

    // Adjust reward
    reward -= 1;

    for (unsigned i = 0; i < num_proc; ++i) {
        float utilization = static_cast<float>(proc_threads[i]) / max_threads;
        if (num_tasks > 0 && utilization < 0.2f) {
            reward -= 1;
        }
    }

    // Check done
    done = (num_tasks == 0);
    for (unsigned threads: proc_threads) {
        if (threads > 0) {
            done = false;
            break;
        }
    }

    nextState.push_back(num_tasks);
    nextState.insert(nextState.end(), proc_threads.begin(), proc_threads.end());

    return std::make_tuple(nextState, reward, done);
    */
    std::vector<unsigned> nextState;
    int reward = 0;
    bool done = true;
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
    this->_remainingDurationInQueue -= t->getRemainingDuration();
    this->_taskQueue.pop_front();

    // Processor is at max cap, re-queued back to the task queue
    if (!this->_processors.at(toProc)->queue(t)) {
        this->_taskQueue.push_front(t);
        this->_remainingDurationInQueue += t->getRemainingDuration();
    }
}
