#include "Processor.h"

Processor::Processor(const unsigned maxThread) {
    this->_maxThread = maxThread;
}

bool Processor::queue(const std::shared_ptr<Task> &task) {
    if (task == nullptr) {
        return false;
    }

    if (this->_tasks.size() < this->_maxThread) {
        this->_tasks.push_back(task);
        this->_totalProcessTime += task->getRemainingDuration();

        return true;
    }

    return false;
}

void Processor::tick() {
    for (unsigned i = 0; i < this->_tasks.size(); i++) {
        // Reduce task's duration. If duration is 0 then task is considered done, remove
        this->_tasks[i]->tick();
        if (this->_tasks[i]->isFinished()) {
            this->_tasks.erase(_tasks.begin() + i);
            --i;
        }

        // Reduce total process time
        if (this->_totalProcessTime != 0) {
            this->_totalProcessTime -= 1;
        }
    }
}

unsigned Processor::getTotalProcessTime() const {
    return this->_totalProcessTime;
}

float Processor::getUtilization() const {
    return static_cast<float>(this->_tasks.size()) / static_cast<float>(this->_maxThread);
}

unsigned Processor::getNumBusyThread() const {
    return this->_tasks.size();
}


