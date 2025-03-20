#include "Processor.h"

Processor::Processor(const unsigned maxCapacity) {
    this->_maxCapacity = maxCapacity;
}

bool Processor::queue(const std::shared_ptr<Task> &task) {
    if (task == nullptr) {
        return false;
    }

    const unsigned newTotalProcessTime = this->_totalProcessTime + task->getRemainingDuration();
    if (newTotalProcessTime < this->_maxCapacity) {
        this->_tasks.push_back(task);
        this->_totalProcessTime = newTotalProcessTime;

        return true;
    }

    return false;
}

void Processor::tick() {
    if (!this->_tasks.empty()) {
        const std::shared_ptr<Task> nextTask = this->_tasks.front();

        // Reduce first task's duration. If duration is 0 then task is considered done, remove
        nextTask->tick();
        if (nextTask->isFinished()) {
            _tasks.erase(_tasks.begin());
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
    return static_cast<float>(this->_totalProcessTime) / static_cast<float>(this->_maxCapacity);
}
