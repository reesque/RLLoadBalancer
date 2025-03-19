#include "Task.h"

Task::Task(const unsigned length) {
    this->_duration = length;
}

void Task::tick() {
    this->_duration--;
}

bool Task::isFinished() const {
    return this->_duration == 0;
}

unsigned Task::getRemainingDuration() const {
    return this->_duration;
}
